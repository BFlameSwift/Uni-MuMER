# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import editdistance

import torch
from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Optional: official Qwen visual pre-processing (resize, patching …)
try:
    from qwen_vl_utils import vision_process  # noqa: F401 (side-effect import)
except ModuleNotFoundError:
    vision_process = None  # type: ignore

# -----------------------------------------------------------------------------#
# Logging
# -----------------------------------------------------------------------------#
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)

def compute_distance(a: str, b: str) -> int:
    """Return the Levenshtein edit distance between *a* and *b*.

    This is a thin wrapper around ``editdistance.eval`` so that downstream code
    can swap implementations if needed without touching call‑sites.
    """
    return editdistance.eval(a.split(), b.split())

# -----------------------------------------------------------------------------
# Three‑line report 
# -----------------------------------------------------------------------------

def quick_report(results: List[Dict[str, str]]) -> None:
    """Print *exprate*, *error1*, and *error2* on three separate lines.

    Args:
        results: list of dicts produced by the evaluation loop, each containing
            ``gt``  – ground‑truth string
            ``pred`` – predicted string
            other keys are ignored here.

    Definitions (matching common evaluation protocols):
        * exprate – percentage of samples with *distance == 0*
        * error1  – percentage with *distance ≤ 1*
        * error2  – percentage with *distance ≤ 2*
    """
    total = len(results)
    if total == 0:
        for tag in ("exprate", "error1", "error2"):
            print(f"{tag}: 0.00%")
        return

    # Compute distance for each sample only once
    dists = [compute_distance(r["pred"], r["gt"]) for r in results]

    def percentage(count: int) -> str:
        """Helper: format *count / total* as percentage with two decimals."""
        return f"{count / total * 100:.2f}%"

    exprate = percentage(sum(d == 0 for d in dists))
    error1  = percentage(sum(d <= 1 for d in dists))
    error2  = percentage(sum(d <= 2 for d in dists))

    # Three‑line summary (order matters!)
    print(f"exprate: {exprate}")
    print(f"error1:  {error1}")
    print(f"error2:  {error2}")


# -----------------------------------------------------------------------------#
# Helper functions
# -----------------------------------------------------------------------------#



def format_chatml(
    tokenizer: AutoTokenizer, user_msg: str, system_msg: str | None = None
) -> str:
    """Construct a Qwen ChatML prompt that embeds an image placeholder."""
    system_msg = system_msg or "You are a helpful assistant."
    placeholder = "<|image_pad|>"

    # We call apply_chat_template only to guarantee correct formatting;
    # its output is then largely overwritten.
    _ = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )

    chatml = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|>{placeholder}"
        f"<|vision_end|>{user_msg}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return chatml


def run_inference(
    model_name: str | Path,
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    suffix: str = "_pred",
    max_tokens: int = 128,
    temperature: float = 0.2,
    top_p: float = 0.8,
) -> None:
    """
    Iterate over every ``*.json`` in ``input_dir`` and write predictions
    to ``output_dir``.  Parameters not supplied fall back to defaults.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Device topology ----------------------------------------------------#
    gpu_cnt = torch.cuda.device_count()
    tp_size = max(gpu_cnt, 1)
    LOGGER.info("%d GPU(s) detected → tensor_parallel_size=%d", gpu_cnt, tp_size)

    # 2) Model & tokenizer --------------------------------------------------#
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,  # required for Qwen
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 1},
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, use_fast=False
    )
    sampling_params = SamplingParams(
        max_tokens=max_tokens, temperature=temperature, top_p=top_p
    )

    # 3) File loop ----------------------------------------------------------#
    for json_path in sorted(input_dir.glob("*.json")):
        LOGGER.info("[FILE] %s", json_path.name)
        with json_path.open(encoding="utf-8") as fp:
            dataset: List[Dict] = json.load(fp)

        requests: List[Dict] = []
        metas: List[Dict] = []

        for record in dataset:
            if not record.get("images"):
                continue
            image_path = record["images"][0]

            prompt_text = gt_text = None
            for msg in record.get("messages", []):
                if msg["from"] == "human":
                    prompt_text = msg["value"].strip()
                elif msg["from"] == "gpt":
                    gt_text = msg["value"].strip()
            if not prompt_text or gt_text is None:
                continue

            chatml = format_chatml(tokenizer, prompt_text)
            try:
                req = {
                    "prompt": chatml,
                    "multi_modal_data": {
                        "image": Image.open(image_path).convert("RGB")
                    },
                }
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("Failed to load %s – %s", image_path, exc)
                continue

            requests.append(req)
            metas.append({"gt": gt_text, "image_path": image_path})

        if not requests:
            LOGGER.info("↳ No valid sample in %s – skipped.", json_path.name)
            continue

        outputs = llm.generate(requests, sampling_params)

        results: List[Dict] = []
        for meta, out in zip(metas, outputs, strict=True):
            results.append(
                {
                    "gt": meta["gt"],
                    "pred": out.outputs[0].text.strip(),
                    "image_path": meta["image_path"],
                    "img_id": Path(meta["image_path"]).stem,
                }
            )
        LOGGER.info("↳ Generated %d records", len(results))
        quick_report(results)
        # 4) Save results -----------------------------------------------------#
        out_file = output_dir / f"{json_path.stem}{suffix}.json"
        out_file.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        LOGGER.info("↳ Saved %d records → %s", len(results), out_file)


# -----------------------------------------------------------------------------#
# CLI
# -----------------------------------------------------------------------------#
def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch multimodal inference with Qwen-2.5-VL-3B + vLLM"
    )
    parser.add_argument(
        "--model",
        default="/home/liyu/workspaces/llama-factory/saves/qwen2.5_vl-3b/full/sft/standred/05x-all-methods/0504_crohme+2023+rewise_hme100k+rewise_mathwriting+im2latex_bs512_+3methods_1epoch_1/checkpoint-3151",
        help="Model path or Hugging Face repo (default: the large local checkpoint).",
    )
    parser.add_argument(
        "--input-dir",
        default="./data/",
        help="Directory containing source JSON files. (default: ./data/)",
    )
    parser.add_argument(
        "--output-dir",
        default="vl_outputs",
        help="Directory to write prediction JSON files. (default: vl_outputs_32b)",
    )
    parser.add_argument(
        "--suffix",
        default="_pred",
        help='Suffix appended to output files (default: "_pred").',
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate (default: 128).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.7,
        help="Nucleus sampling top-p (default: 0.8).",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover
    args = parse_args()
    print("Running inference with the following parameters:")
    print(f"Model: {args.model}")
    print(f"Input directory: {args.input_dir}")
    run_inference(
        model_name=args.model,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        suffix=args.suffix,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()