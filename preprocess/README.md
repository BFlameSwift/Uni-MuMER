# Preprocessing Scripts

We provide the preprocessing utilities used to convert the original notebook
workflows into reusable command-line tools for dataset construction and
reproducibility.

## MathNet4clean Submodule

We track [`preprocess/MathNet4clean`](./MathNet4clean) as a git submodule to
keep the external MathNet preprocessing code reproducible and versioned
separately from the Uni-MuMER codebase.

Initialize the submodule after cloning the main repository:

```bash
git submodule update --init --recursive
```

On top of the cloned submodule, we port the archived Uni-MuMER MathNet cleaner
as:

- `preprocess/MathNet4clean/preprocessing/improve_tokens_unimumer.py`
- `preprocess/MathNet4clean/preprocessing/latex_cate_lib.py`

This path is wired into
`preprocess/MathNet4clean/tools/dataloader.py` and keeps the original upstream
`preprocessing/improve_tokens.py` intact for reference.

## MathNet HMER Data Generation

`make_mathnet_hmer_data.py` ports the archived notebook
`repo/mathnet-ly/0410_proc_file.ipynb` into a reusable CLI.

It provides two subcommands:

- `process`: run the full MathNet tokenize/normalize pipeline and export the
  intermediate files, invalid-record log, and final HMER prompt JSON
- `prompt-only`: build prompt JSON directly from original captions without the
  MathNet cleaning stages

The `process` subcommand keeps the original notebook-style artifact layout,
including:

- `step1_csv.csv`
- `step1.5_csv.csv`
- `step2_captions.txt`
- `step3_tokenized.txt`
- `step4_restored.csv`
- `step4_restored_normalized.csv`
- `step6_final.txt`
- `step7_final.json`
- `step7_point_five_caption_only.txt`
- `step7_point_six_tokenize_txt_only.txt`
- `step7_point_seven_caption_json.json`
- `step8_post_delete.json`
- `step9_hmer_json.json`
- `invalid_record.json`
- `<output_dir_name>_0425_final.json`

Compared with the notebook, the CLI makes a few intentional engineering
improvements while preserving the active data flow:

- it uses the in-repo `preprocess/MathNet4clean` submodule instead of hard-coded
  external paths
- it calls `improve_tokens_unimumer.py`, which is the cleaned port of the
  archived `improve_tokens_ly.py`
- it supports both `.txt` and `.json` inputs via explicit CLI arguments
- it provides reproducible sample inputs and committed sample outputs

Example:

```bash
python preprocess/make_mathnet_hmer_data.py process \
  --input preprocess/examples/mathnet_hmer/sample_caption.txt \
  --output-dir preprocess/examples/mathnet_hmer/processed_output \
  --image-base-path preprocess/examples/mathnet_hmer/images \
  --image-suffix .png \
  --mathnet-root preprocess/MathNet4clean
```

Ready-to-run examples are provided in `preprocess/examples/mathnet_hmer/`.

## UniMuMER-Tree Data Generation

`make_unimumer_tree_data.py` ports the current source notebook
`notebook/tdv2/0425_make_tdv2_data.ipynb` into a reusable CLI.

The tree serializer is kept behaviorally consistent with the source notebook so
that the generated tree strings match the original preprocessing output for the
same tokenized LaTeX input.

Supported tasks:

- `tree`
- `can`

Input format:

- a JSON array
- each record contains `images` and `messages`
- `messages[0]` is the user prompt, `messages[1]` is the target LaTeX

Note: backslashes are escaped in JSON strings. For example, the LaTeX token `\\`
is stored as `\\\\` in the JSON text, but it is still counted as the token `\\`.

Examples:

```bash
python preprocess/make_unimumer_tree_data.py \
  --task tree \
  --input /path/to/crohme_train.json \
  --output /path/to/crohme_train_tree.json
```

```bash
python preprocess/make_unimumer_tree_data.py \
  --task can \
  --input /path/to/train_json_dir \
  --output /path/to/output_dir \
  --include-substring train \
  --exclude-substring original
```

Example files for reproducibility are provided in
`preprocess/examples/unimumer_tree/`.

## Acknowledgements

The tree-related code is adapted and refined with reference to:

- TDv2: https://github.com/yqingli123/TDv2/blob/6da7bf9e33b687af585f6b98b6a4b41a50fdb1e8/utils/latex2gtd_v2_2.py
- TAMER: https://github.com/qingzhenduyu/TAMER/blob/main/tamer/datamodule/latex2gtd.py

Compared with the referenced upstream files, this local implementation keeps
the original notebook tree-string output while making the codebase easier to
reuse in preprocessing scripts:

- It preserves the source notebook tokenization path so generated tree strings
  stay consistent with the original UniMuMER-Tree preprocessing results.
- It supports generic split environment tokens such as `\begin {cases}` and
  `\end {cases}`, while TDv2 and TAMER mainly target the older matrix-style
  environment tokens such as `\begin{matrix}`.
- It moves operator categories into `preprocess/unimumer_tree/latex_cate_lib.py`
  instead of hard-coding every command list directly inside one parser file.
