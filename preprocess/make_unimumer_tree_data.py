"""CLI for generating UniMuMER-Tree supervision data from ShareGPT JSON."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

try:
    from preprocess.unimumer_tree.latex2tree import latex_to_tree_str
except ModuleNotFoundError:
    from unimumer_tree.latex2tree import latex_to_tree_str

TREE_PROMPT = (
    "I have an image of a handwritten mathematical expression. Please generate "
    "the abstract syntax tree (AST) of the formula in the image, and then "
    "provide its corresponding LaTeX format."
)
CAN_PROMPT = (
    "I have an image of a handwritten mathematical expression. Please identify "
    "and count each distinct visible mathematical symbol in the image, and then "
    "provide its corresponding LaTeX format."
)

TASK_SUFFIXES = {
    "tree": "_tree",
    "can": "_can",
}
SUPPORTED_TASKS = tuple(TASK_SUFFIXES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate UniMuMER-Tree style training JSON files from ShareGPT-style source JSON."
    )
    parser.add_argument("--task", choices=SUPPORTED_TASKS, required=True, help="Task type to generate.")
    parser.add_argument("--input", required=True, help="Input JSON file or directory.")
    parser.add_argument(
        "--output",
        help="Output JSON file or directory. If omitted, outputs are derived from the input path.",
    )
    parser.add_argument("--prefix", default="", help="Optional filename prefix for generated files.")
    parser.add_argument(
        "--suffix",
        help="Optional filename suffix. Defaults to the notebook-compatible suffix for the selected task.",
    )
    parser.add_argument(
        "--include-substring",
        help="When input is a directory, only process files whose names contain this substring.",
    )
    parser.add_argument(
        "--exclude-substring",
        help="When input is a directory, skip files whose names contain this substring.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on the first malformed record instead of skipping it.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=4,
        help="Indent level for generated JSON. Default: 4.",
    )
    return parser.parse_args()


def get_symbol_counts(latex_string: str) -> dict[str, int]:
    # Preserve the first-seen token order so the output stays stable across runs.
    count_dict: dict[str, int] = {}
    for token in latex_string.split():
        count_dict[token] = count_dict.get(token, 0) + 1
    count_dict.pop("{", None)
    count_dict.pop("}", None)
    return count_dict


def make_count_dict2str(count_dict: dict[str, int]) -> str:
    return ", ".join(f"{key}: {value}" for key, value in count_dict.items())


def validate_record(record: dict[str, Any]) -> tuple[str, str]:
    try:
        images = record["images"]
        messages = record["messages"]
        image_path = images[0]
        latex = messages[1]["value"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("record must contain images[0] and messages[1].value") from exc

    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("messages must contain at least two items")
    if not isinstance(messages[0], dict) or not isinstance(messages[1], dict):
        raise ValueError("messages[0] and messages[1] must be objects")
    if not isinstance(image_path, str):
        raise ValueError("images[0] must be a string")
    if not isinstance(latex, str):
        raise ValueError("messages[1].value must be a string for this conversion")
    return image_path, latex


def make_tree_record(record: dict[str, Any]) -> dict[str, Any]:
    new_record = copy.deepcopy(record)
    _, latex = validate_record(new_record)
    # Keep the original ShareGPT record layout and only replace the prompt/answer payload.
    new_record["messages"][0]["value"] = f"<image>{TREE_PROMPT}"
    new_record["messages"][1]["value"] = f"{latex_to_tree_str(latex)}\n{latex}"
    return new_record


def make_can_record(record: dict[str, Any]) -> dict[str, Any]:
    image_path, latex = validate_record(record)
    symbol_counts = get_symbol_counts(latex)
    value: Any = f"{make_count_dict2str(symbol_counts)}\n{latex}"

    return {
        "images": [image_path],
        "messages": [
            {"from": "human", "value": f"<image>{CAN_PROMPT}"},
            {"from": "gpt", "value": value},
        ],
    }


def convert_record(record: dict[str, Any], task: str, _index: int) -> dict[str, Any]:
    if task == "tree":
        return make_tree_record(record)
    if task == "can":
        return make_can_record(record)
    raise ValueError(f"unsupported task: {task}")


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return data


def dump_json(path: Path, data: list[dict[str, Any]], indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=indent)


def convert_file(input_path: Path, output_path: Path, task: str, strict: bool, indent: int) -> tuple[int, int]:
    data = load_json(input_path)
    converted: list[dict[str, Any]] = []
    skipped = 0

    for index, record in enumerate(tqdm(data, desc=input_path.name)):
        try:
            converted.append(convert_record(record, task, index))
        except Exception:
            if strict:
                raise
            skipped += 1

    dump_json(output_path, converted, indent)
    return len(converted), skipped


def resolve_output_file(input_file: Path, output_root: Path | None, task: str, prefix: str, suffix: str | None) -> Path:
    resolved_suffix = TASK_SUFFIXES[task] if suffix is None else suffix
    filename = f"{prefix}{input_file.stem}{resolved_suffix}{input_file.suffix}"

    if output_root is None:
        return input_file.with_name(filename)

    # A missing path without a JSON suffix is treated as an output directory.
    if output_root.exists() and output_root.is_dir():
        return output_root / filename

    if not output_root.exists() and output_root.suffix != input_file.suffix:
        return output_root / filename

    return output_root


def iter_input_files(input_path: Path, include_substring: str | None, exclude_substring: str | None) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"input path not found: {input_path}")

    candidates = sorted(path for path in input_path.iterdir() if path.is_file() and path.suffix == ".json")
    results: list[Path] = []
    for path in candidates:
        if include_substring and include_substring not in path.name:
            continue
        if exclude_substring and exclude_substring in path.name:
            continue
        results.append(path)
    return results


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_root = Path(args.output) if args.output else None

    if input_path.is_dir() and output_root is not None:
        if output_root.exists() and output_root.is_file():
            raise ValueError("when --input is a directory, --output must be a directory or be omitted")
        if not output_root.exists() and output_root.suffix == ".json":
            raise ValueError("when --input is a directory, --output cannot be a single .json file path")

    input_files = iter_input_files(input_path, args.include_substring, args.exclude_substring)

    if not input_files:
        raise FileNotFoundError("no matching input JSON files were found")

    for input_file in input_files:
        output_file = resolve_output_file(input_file, output_root, args.task, args.prefix, args.suffix)
        written, skipped = convert_file(input_file, output_file, args.task, args.strict, args.indent)
        print(f"{input_file} -> {output_file} | kept={written} skipped={skipped}")


if __name__ == "__main__":
    main()
