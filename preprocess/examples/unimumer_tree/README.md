# UniMuMER-Tree Examples

We provide a minimal ShareGPT-style example together with the corresponding
generated outputs for the UniMuMER-Tree preprocessing script.

The bundled image is a repository-local placeholder used to demonstrate the
expected data schema and preprocessing outputs. The example targets are
illustrative and are intended for format verification rather than semantic
evaluation.

Files:

- `sample_input.json`: source data before conversion
- `sample_tree.json`: tree supervision output
- `sample_can.json`: symbol counting output with LaTeX target

Note: backslashes are escaped in JSON strings. For example, the LaTeX token `\\`
is stored as `\\\\` in the JSON text, but it is still counted as the token `\\`.

The following commands reproduce the released example outputs:

```bash
python preprocess/make_unimumer_tree_data.py \
  --task tree \
  --input preprocess/examples/unimumer_tree/sample_input.json \
  --output preprocess/examples/unimumer_tree/sample_tree.json
```

```bash
python preprocess/make_unimumer_tree_data.py \
  --task can \
  --input preprocess/examples/unimumer_tree/sample_input.json \
  --output preprocess/examples/unimumer_tree/sample_can.json
```
