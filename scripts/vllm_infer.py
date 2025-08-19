# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import editdistance
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams

# Optional: official Qwen visual pre-processing (resize, patching …)
try:
    from qwen_vl_utils import (
        process_vision_info,
        vision_process,  # noqa: F401 (side-effect import)
    )
    from qwen_vl_utils.vision_process import fetch_image
except ModuleNotFoundError:
    vision_process = None  # type: ignore
    
from eval_metrics_calculator import evaluate_text_generation

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
    max_tokens: int = 2048,
    temperature: float = 0.95,
    top_p: float = 0.8,
    top_k: int = 50,
    batch_size: int = 32768,  # vLLM uses dynamic batching; this is just an upper limit
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
        # limit_mm_per_prompt={"image": 1},
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, use_fast=False
    )
    processor = AutoProcessor.from_pretrained( model_name, trust_remote_code=True, use_fast=False)
    sampling_params = SamplingParams(
        max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k,stop=["<|im_end|>", "<|endoftext|>"]
    )

    # 3) File loop ----------------------------------------------------------#
    for json_path in sorted(input_dir.glob("*.json")):
        LOGGER.info("[FILE] %s", json_path.name)
        with json_path.open(encoding="utf-8") as fp:
            dataset: List[Dict] = json.load(fp)

        results: List[Dict] = []
        reqs_batch: List[Dict] = []
        metas_batch: List[Dict] = []

        def flush_batch():
            """Send current batch to vLLM, collect outputs, then free memory."""
            if not reqs_batch:
                return
            outputs = llm.generate(reqs_batch, sampling_params)
            for meta, out in zip(metas_batch, outputs, strict=True):
                results.append(
                    {
                        "gt": meta["gt"],
                        "pred": out.outputs[0].text.strip(),
                        "image_path": meta["image_path"],
                        "img_id": Path(meta["image_path"]).stem,
                    }
                )
            reqs_batch.clear()
            metas_batch.clear()
            del outputs
            # torch.cuda.empty_cache()
            # gc.collect()

        valid_cnt = 0
        for record in tqdm(dataset, desc="Processing records", unit="record"):
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

            image_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt_text},
                    ],
                },
            ]
            final_prompt = processor.apply_chat_template(
                image_messages, tokenize=False, add_generation_prompt=True
            )

            # 仅为当前样本构建/加载多模态输入，避免一次性加载全部图像
            image_inputs, _, _ = process_vision_info(image_messages, return_video_kwargs=True)
            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs

            llm_inputs = {
                "prompt": final_prompt,
                "multi_modal_data": mm_data,
            }

            reqs_batch.append(llm_inputs)
            metas_batch.append({"gt": gt_text, "image_path": image_path})
            valid_cnt += 1

            if len(reqs_batch) >= batch_size:
                LOGGER.info("↳ Flushing a batch of %d (file=%s)", len(reqs_batch), json_path.name)
                flush_batch()

        # 处理剩余未满批的数据
        flush_batch()

        if not results:
            LOGGER.info("↳ No valid sample in %s – skipped.", json_path.name)
            continue

        LOGGER.info("↳ Generated %d records from %s", len(results), json_path.name)

        # 4) Save results -----------------------------------------------------#
        out_file = output_dir / f"{json_path.stem}{suffix}.json"
        out_file.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        LOGGER.info("↳ Saved %d records → %s", len(results), out_file)

        # 5) Evaluate results -------------------------------------------------#
        metrics = evaluate_text_generation(out_file, os.path.join(output_dir, f"{json_path.stem}_results.txt"))


    # exit the context manager to close the vLLM engine

    # import ray
    # ray.shutdown() 

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
        default="",
        help="Model path or Hugging Face repo (default: the large local checkpoint).",
    )
    parser.add_argument(
        "--input-dir",
        default="./data/",
        help="Directory containing source JSON files. (default: ./data/)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
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
        default=2048,
        help="Maximum number of tokens to generate (default: 2048).",
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
        default=0.8,
        help="Nucleus sampling top-p (default: 0.8).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32768,
        help="Number of samples per vLLM.generate() call (default: 32768).",
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
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()