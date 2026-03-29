"""Port of mathnet-ly/0410_proc_file.ipynb into a reusable CLI."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


DEFAULT_HUMAN_PROMPT = (
    "<image>I have an image of a handwritten mathematical expression. "
    "Please write out the expression of the formula in the image using LaTeX format."
)

FIRST_DELETED = [
    "{}",
    r"\underline{\quad}",
    r"\widehat{\}",
    r"\scriptstyle{k-1}\choose{r}",
    r"\cfrac{\cfrac{\cfrac{",
]

CHINESE_INVALID_TOKENS = [
    "、",
    "。",
    "丙",
    "个",
    "为",
    "乙",
    "人",
    "代",
    "以",
    "倍",
    "元",
    "入",
    "公",
    "分",
    "即",
    "原",
    "可",
    "周",
    "和",
    "因",
    "大",
    "天",
    "妈",
    "将",
    "小",
    "岁",
    "年",
    "式",
    "得",
    "或",
    "所",
    "把",
    "数",
    "方",
    "时",
    "明",
    "是",
    "最",
    "本",
    "次",
    "求",
    "法",
    "满",
    "然",
    "甲",
    "的",
    "种",
    "积",
    "第",
    "答",
    "组",
    "能",
    "自",
    "行",
    "袋",
    "解",
    "该",
    "负",
    "足",
    "面",
    "页",
    "项",
    "√",
]

INVALID_SUBSTRINGS = FIRST_DELETED + CHINESE_INVALID_TOKENS


@dataclass
class SampleRecord:
    caption: str
    image_id: str


@dataclass
class PipelinePaths:
    csv_output: Path
    filtered_csv: Path
    caption_txt: Path
    tokenized_txt: Path
    restored_csv: Path
    normalized_csv: Path
    final_caption_txt: Path
    final_caption_json: Path
    caption_only_txt: Path
    normalized_caption_txt: Path
    validated_caption_json: Path
    post_delete_json: Path
    hmer_json: Path
    invalid_record_json: Path
    final_copy_json: Path


def parse_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def run_command(argv: list[str], cwd: Path, show_output: bool = False) -> None:
    start_time = time.time()
    if show_output:
        completed = subprocess.run(argv, cwd=cwd, check=False)
    else:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    duration = time.time() - start_time
    print(f"Command: {' '.join(argv)}")
    print(f"Return code: {completed.returncode}")
    print(f"Execution time: {duration:.2f} seconds")
    if completed.returncode != 0:
        if not show_output:
            if completed.stdout:
                print(completed.stdout)
            if completed.stderr:
                print(completed.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed with return code {completed.returncode}")


def check_brackets(formula_str: str) -> bool:
    stack: list[str] = []
    pairs = {")": "(",
        "]": "[",
        "}": "{",
    }
    for token in formula_str.split():
        if token in ["(", "[", "{"]:
            stack.append(token)
        elif token in pairs:
            if not stack:
                return False
            top = stack.pop()
            if pairs[token] != top:
                return False
    return len(stack) == 0


def is_invalid_caption(caption: str) -> bool:
    compact = "".join(str(caption).strip().split())
    return any(item in compact for item in INVALID_SUBSTRINGS)


def load_records(
    input_path: Path,
    json_image_id_key: str,
    json_caption_key: str,
) -> list[SampleRecord]:
    if input_path.suffix == ".json":
        data = json.loads(input_path.read_text())
        return [
            SampleRecord(str(item[json_caption_key]), str(item[json_image_id_key]))
            for item in data
        ]
    records = []
    for raw_line in input_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        records.append(SampleRecord(" ".join(parts[1:]), parts[0]))
    return records


def write_caption_csv(records: list[SampleRecord], output_path: Path) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["caption", "image_id"])
        for record in records:
            writer.writerow([record.caption, record.image_id])


def read_caption_csv(csv_path: Path) -> list[SampleRecord]:
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        return [SampleRecord(row["caption"], row["image_id"]) for row in reader]


def read_plain_csv(csv_path: Path) -> list[SampleRecord]:
    with csv_path.open() as f:
        reader = csv.reader(f)
        return [SampleRecord(row[0], row[1]) for row in reader]


def write_plain_csv(records: list[SampleRecord], output_path: Path) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        for record in records:
            writer.writerow([record.caption, record.image_id])


def image_path_for(image_base_path: str, image_id: str, image_suffix: str) -> str:
    return str(Path(image_base_path) / f"{image_id}{image_suffix}")


def record_invalid(
    invalid_records: list[dict[str, str]],
    image_id: str,
    caption: str,
    image_base_path: str,
    image_suffix: str,
    invalid_type: str,
) -> None:
    invalid_records.append(
        {
            "image_id": image_id,
            "caption": caption,
            "image_path": image_path_for(image_base_path, image_id, image_suffix),
            "type": invalid_type,
        }
    )


def filter_invalid_caption_csv(
    input_csv: Path,
    output_csv: Path,
    invalid_records: list[dict[str, str]],
    image_base_path: str,
    image_suffix: str,
) -> None:
    kept: list[SampleRecord] = []
    removed = 0
    for record in read_caption_csv(input_csv):
        if is_invalid_caption(record.caption):
            removed += 1
            record_invalid(
                invalid_records,
                record.image_id,
                record.caption,
                image_base_path,
                image_suffix,
                "1.5invalid_delete",
            )
            continue
        kept.append(record)
    print(f"one_point_five_filter_invalid_caption: {removed}")
    write_caption_csv(kept, output_csv)


def write_caption_txt_only(records: list[SampleRecord], output_path: Path) -> None:
    with output_path.open("w") as f:
        for record in records:
            f.write(f"{record.caption}\n")


def run_mathnet_preprocess(
    mathnet_root: Path,
    input_path: Path,
    output_path: Path,
    mode: str,
    show_output: bool = False,
) -> None:
    run_command(
        [
            sys.executable,
            "preprocessing/preprocess_formulas.py",
            "--input-file",
            str(input_path.resolve()),
            "--output-file",
            str(output_path.resolve()),
            "--mode",
            mode,
        ],
        cwd=mathnet_root,
        show_output=show_output,
    )


def restore_csv_from_tokenized(
    tokenize_path: Path,
    original_csv: Path,
    output_path: Path,
    invalid_records: list[dict[str, str]],
    image_base_path: str,
    image_suffix: str,
) -> None:
    tokenize_data = tokenize_path.read_text().splitlines()
    original_records = read_caption_csv(original_csv)
    restored: list[SampleRecord] = []
    for idx, item in enumerate(tokenize_data):
        original = original_records[idx]
        if not check_brackets(item):
            record_invalid(
                invalid_records,
                original.image_id,
                original.caption,
                image_base_path,
                image_suffix,
                "4_tokenization_error",
            )
            continue
        if item != "":
            restored.append(SampleRecord(item, original.image_id))
        else:
            record_invalid(
                invalid_records,
                original.image_id,
                original.caption,
                image_base_path,
                image_suffix,
                "4_tokenization_error",
            )
    write_plain_csv(restored, output_path)


def run_mathnet_normalization(
    mathnet_root: Path,
    csv_path: Path,
    use_only_c: bool,
    remove: bool,
    show_output: bool = True,
) -> Path:
    file_ending = "_normalized.csv"
    run_command(
        [
            sys.executable,
            "preprocessing/improve_tokens_unimumer.py",
            "--files",
            str(csv_path.resolve()),
            "--file-ending",
            file_ending,
            "--use-only-c",
            str(use_only_c),
            "--remove",
            str(remove),
        ],
        cwd=mathnet_root,
        show_output=show_output,
    )
    return csv_path.with_name(csv_path.stem + file_ending)


def collect_normalization_errors(
    original_csv_path: Path,
    normalized_csv_path: Path,
    invalid_records: list[dict[str, str]],
    image_base_path: str,
    image_suffix: str,
) -> None:
    original_records = read_plain_csv(original_csv_path)
    normalized_records = read_plain_csv(normalized_csv_path)
    i, j = 0, 0
    catch_count = 0
    while i < len(original_records) and j < len(normalized_records):
        if original_records[i].image_id != normalized_records[j].image_id:
            record_invalid(
                invalid_records,
                original_records[i].image_id,
                original_records[i].caption,
                image_base_path,
                image_suffix,
                "5.5_normalize_error",
            )
            i += 1
            catch_count += 1
        else:
            i += 1
            j += 1
    while i < len(original_records):
        record_invalid(
            invalid_records,
            original_records[i].image_id,
            original_records[i].caption,
            image_base_path,
            image_suffix,
            "5.5_normalize_error",
        )
        i += 1
    print(f"catch_cnt: {catch_count}")
    print(len(original_records) - len(normalized_records))


def make_caption_txt_from_csv(csv_path: Path, output_path: Path) -> None:
    records = read_plain_csv(csv_path)
    with output_path.open("w") as f:
        for record in records:
            f.write(f"{record.image_id} {record.caption}\n")


def make_caption_json(
    caption_path: Path,
    output_path: Path,
    image_base_path: str,
    image_suffix: str,
) -> None:
    output_data = []
    for raw_line in caption_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        image_id = parts[0]
        caption = " ".join(parts[1:])
        output_data.append(
            {
                "image_path": image_path_for(image_base_path, image_id, image_suffix),
                "caption": caption,
                "image_id": image_id,
            }
        )
    output_path.write_text(json.dumps(output_data, indent=4, ensure_ascii=False))


def write_caption_only_from_caption_json(caption_json_path: Path, output_path: Path) -> None:
    data = json.loads(caption_json_path.read_text())
    with output_path.open("w") as f:
        for item in data:
            f.write(f"{item['caption']}\n")


def validate_caption_json_with_normalized_lines(
    normalized_caption_txt: Path,
    past_caption_json_path: Path,
    output_path: Path,
    invalid_records: list[dict[str, str]],
    image_base_path: str,
    image_suffix: str,
) -> None:
    lines = normalized_caption_txt.read_text().splitlines()
    data = json.loads(past_caption_json_path.read_text())
    output_data = []
    empty_count = 0
    for idx, line in enumerate(lines):
        current = data[idx]
        if not check_brackets(line):
            record_invalid(
                invalid_records,
                current["image_id"],
                current["caption"],
                image_base_path,
                image_suffix,
                "7.7_invalid_bracket",
            )
            continue
        if line.strip() == "":
            empty_count += 1
            record_invalid(
                invalid_records,
                current["image_id"],
                current["caption"],
                image_base_path,
                image_suffix,
                "7.7_empty_line",
            )
            continue
        output_data.append(
            {
                "image_path": current["image_path"],
                "caption": current["caption"],
                "image_id": current["image_id"],
            }
        )
    print(f"empty_cnt: {empty_count}")
    output_path.write_text(json.dumps(output_data, indent=4, ensure_ascii=False))


def post_delete_json(
    json_path: Path,
    output_path: Path,
    delete_keys: list[str],
    invalid_records: list[dict[str, str]],
    image_base_path: str,
    image_suffix: str,
) -> None:
    output_data = []
    data = json.loads(json_path.read_text())
    for item in data:
        ok = True
        for token in item["caption"].split():
            if token in delete_keys:
                ok = False
                break
        if ok:
            output_data.append(item)
        else:
            record_invalid(
                invalid_records,
                item["image_id"],
                item["caption"],
                image_base_path,
                image_suffix,
                "8_post_delete",
            )
    output_path.write_text(json.dumps(output_data, indent=4, ensure_ascii=False))


def generate_prompt_record(image_path: str, caption: str) -> dict[str, object]:
    return {
        "images": [image_path],
        "messages": [
            {
                "from": "human",
                "value": DEFAULT_HUMAN_PROMPT,
            },
            {
                "from": "gpt",
                "value": caption,
            },
        ],
    }


def make_hmer_json_format(json_path: Path, output_path: Path) -> None:
    data = json.loads(json_path.read_text())
    output_data = [
        generate_prompt_record(item["image_path"], item["caption"]) for item in data
    ]
    output_path.write_text(json.dumps(output_data, indent=4, ensure_ascii=False))


def build_pipeline_paths(output_dir: Path, final_name_suffix: str) -> PipelinePaths:
    final_copy_name = output_dir.name + final_name_suffix
    return PipelinePaths(
        csv_output=output_dir / "step1_csv.csv",
        filtered_csv=output_dir / "step1.5_csv.csv",
        caption_txt=output_dir / "step2_captions.txt",
        tokenized_txt=output_dir / "step3_tokenized.txt",
        restored_csv=output_dir / "step4_restored.csv",
        normalized_csv=output_dir / "step4_restored_normalized.csv",
        final_caption_txt=output_dir / "step6_final.txt",
        final_caption_json=output_dir / "step7_final.json",
        caption_only_txt=output_dir / "step7_point_five_caption_only.txt",
        normalized_caption_txt=output_dir / "step7_point_six_tokenize_txt_only.txt",
        validated_caption_json=output_dir / "step7_point_seven_caption_json.json",
        post_delete_json=output_dir / "step8_post_delete.json",
        hmer_json=output_dir / "step9_hmer_json.json",
        invalid_record_json=output_dir / "invalid_record.json",
        final_copy_json=output_dir / final_copy_name,
    )


def process_dataset(args: argparse.Namespace) -> None:
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    mathnet_root = Path(args.mathnet_root).resolve()

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    invalid_records: list[dict[str, str]] = []
    paths = build_pipeline_paths(output_dir, args.final_name_suffix)

    print("start processing")
    original_records = load_records(
        input_path,
        json_image_id_key=args.json_image_id_key,
        json_caption_key=args.json_caption_key,
    )
    print(f"original caption lines count: {len(original_records)}")

    write_caption_csv(original_records, paths.csv_output)
    print("step1 done")

    filter_invalid_caption_csv(
        paths.csv_output,
        paths.filtered_csv,
        invalid_records,
        args.image_base_path,
        args.image_suffix,
    )
    print("step1.5 done")

    write_caption_txt_only(read_caption_csv(paths.filtered_csv), paths.caption_txt)
    print("step2 done")

    run_mathnet_preprocess(
        mathnet_root,
        paths.caption_txt,
        paths.tokenized_txt,
        mode="tokenize",
        show_output=args.show_command_output,
    )
    print("step3 done")

    restore_csv_from_tokenized(
        paths.tokenized_txt,
        paths.filtered_csv,
        paths.restored_csv,
        invalid_records,
        args.image_base_path,
        args.image_suffix,
    )
    print("step4 done")

    normalized_path = run_mathnet_normalization(
        mathnet_root,
        paths.restored_csv,
        use_only_c=args.use_only_c,
        remove=args.remove,
        show_output=args.show_command_output,
    )
    if normalized_path != paths.normalized_csv:
        shutil.move(normalized_path, paths.normalized_csv)
    print("step5 done")

    collect_normalization_errors(
        paths.restored_csv,
        paths.normalized_csv,
        invalid_records,
        args.image_base_path,
        args.image_suffix,
    )

    make_caption_txt_from_csv(paths.normalized_csv, paths.final_caption_txt)
    print("step6 done")

    make_caption_json(
        paths.final_caption_txt,
        paths.final_caption_json,
        args.image_base_path,
        args.image_suffix,
    )
    print("step7 done")

    write_caption_only_from_caption_json(paths.final_caption_json, paths.caption_only_txt)
    print("step7.5 done")

    run_mathnet_preprocess(
        mathnet_root,
        paths.caption_only_txt,
        paths.normalized_caption_txt,
        mode="normalize",
        show_output=args.show_command_output,
    )
    print("step7.6 done")

    validate_caption_json_with_normalized_lines(
        paths.normalized_caption_txt,
        paths.final_caption_json,
        paths.validated_caption_json,
        invalid_records,
        args.image_base_path,
        args.image_suffix,
    )
    print("step7.7 done")

    post_delete_json(
        paths.validated_caption_json,
        paths.post_delete_json,
        delete_keys=args.delete_token,
        invalid_records=invalid_records,
        image_base_path=args.image_base_path,
        image_suffix=args.image_suffix,
    )
    print("step8 done")

    make_hmer_json_format(paths.post_delete_json, paths.hmer_json)
    print("step9 done")

    final_count = len(json.loads(paths.hmer_json.read_text()))
    print("all done")
    print(f"final caption lines count: {final_count}")
    print("loss caption lines count: ", len(original_records) - final_count)

    paths.invalid_record_json.write_text(
        json.dumps(invalid_records, indent=4, ensure_ascii=False)
    )
    print(f"invalid data count: {len(invalid_records)}")
    for key, value in Counter(item["type"] for item in invalid_records).items():
        print(f"{key}: {value}")

    shutil.copy(paths.hmer_json, paths.final_copy_json)


def make_json_from_original_caption(
    input_path: Path,
    output_path: Path,
    image_base_path: str,
    image_suffix: str,
    json_image_id_key: str,
    json_caption_key: str,
) -> None:
    records = load_records(input_path, json_image_id_key, json_caption_key)
    output_data = [
        {
            "image_path": image_path_for(image_base_path, record.image_id, image_suffix),
            "caption": record.caption,
            "image_id": record.image_id,
        }
        for record in records
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_data, indent=4, ensure_ascii=False))


def make_prompt_json_format(json_path: Path, output_dir: Path) -> Path:
    data = json.loads(json_path.read_text())
    output_data = [
        generate_prompt_record(item["image_path"], item["caption"]) for item in data
    ]
    output_path = output_dir / f"{output_dir.name}.json"
    output_path.write_text(json.dumps(output_data, indent=4, ensure_ascii=False))
    return output_path


def prompt_only_from_original_caption(args: argparse.Namespace) -> None:
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mid_output_path = output_dir / "mid_output.json"
    make_json_from_original_caption(
        input_path,
        mid_output_path,
        args.image_base_path,
        args.image_suffix,
        args.json_image_id_key,
        args.json_caption_key,
    )
    final_output_path = make_prompt_json_format(mid_output_path, output_dir)
    print(f"prompt-only output: {final_output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Port of mathnet-ly/0410_proc_file.ipynb into a reusable CLI."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    process_parser = subparsers.add_parser(
        "process",
        help="Run the full MathNet-based caption cleaning and HMER JSON pipeline.",
    )
    process_parser.add_argument("--input", required=True, help="Input .txt or .json caption file")
    process_parser.add_argument("--output-dir", required=True, help="Output directory for all pipeline artifacts")
    process_parser.add_argument("--image-base-path", required=True, help="Base directory for image paths written to JSON")
    process_parser.add_argument("--image-suffix", default=".jpg", help="Image suffix appended to image_id")
    process_parser.add_argument(
        "--mathnet-root",
        default=str(Path(__file__).resolve().parent / "MathNet4clean"),
        help="Path to the MathNet4clean submodule root",
    )
    process_parser.add_argument(
        "--json-image-id-key",
        default="img_id",
        help="Image id key for JSON inputs",
    )
    process_parser.add_argument(
        "--json-caption-key",
        default="label",
        help="Caption key for JSON inputs",
    )
    process_parser.add_argument(
        "--use-only-c",
        type=parse_bool,
        default=True,
        help='Whether to rewrite array alignment tokens to "c" during normalization',
    )
    process_parser.add_argument(
        "--remove",
        type=parse_bool,
        default=True,
        help="Whether to remove formulas that fail normalization checks",
    )
    process_parser.add_argument(
        "--delete-token",
        action="append",
        default=[],
        help="Caption token to drop in the post-delete stage; may be repeated",
    )
    process_parser.add_argument(
        "--final-name-suffix",
        default="_0425_final.json",
        help="Suffix for the copied final JSON file in the output directory",
    )
    process_parser.add_argument(
        "--show-command-output",
        action="store_true",
        help="Stream output from MathNet subprocesses instead of capturing it",
    )
    process_parser.set_defaults(func=process_dataset)

    prompt_only_parser = subparsers.add_parser(
        "prompt-only",
        help="Build prompt JSON directly from original captions without MathNet cleaning.",
    )
    prompt_only_parser.add_argument("--input", required=True, help="Input .txt or .json caption file")
    prompt_only_parser.add_argument("--output-dir", required=True, help="Output directory")
    prompt_only_parser.add_argument("--image-base-path", required=True, help="Base directory for image paths written to JSON")
    prompt_only_parser.add_argument("--image-suffix", default="", help="Image suffix appended to image_id")
    prompt_only_parser.add_argument(
        "--json-image-id-key",
        default="img_id",
        help="Image id key for JSON inputs",
    )
    prompt_only_parser.add_argument(
        "--json-caption-key",
        default="label",
        help="Caption key for JSON inputs",
    )
    prompt_only_parser.set_defaults(func=prompt_only_from_original_caption)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
