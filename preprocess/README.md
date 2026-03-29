# Preprocessing Scripts

We provide the preprocessing utilities used to convert the original notebook
workflows into reusable command-line tools for dataset construction and
reproducibility.

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
