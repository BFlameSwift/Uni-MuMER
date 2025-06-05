
# Im2LaTeXv2 Dataset

## Overview

The Im2LaTeXv2 dataset is specifically designed for evaluating mathematical expression recognition systems. This dataset provides both original and normalized test sets. The normalization is based on MathNet standards with additional refinements:

* Standardized token forms (`\le` → `\leq`, `\ge` → `\geq`).
* Converted `\dots` to `\ldots`.
* Added curly braces for clarity (`a_i` → `a_{i}`).
* Adjusted spacing in environments (`\begin{matrix}` → `\begin {matrix}`).

## Dataset Structure

```text
Im2LaTeXv2
├── prompts
│   ├── im2latexv2_test.json
│   └── im2latexv2_test_normalized.json
├── results
│   ├── im2latexv2_test_normalized_pred.json
│   ├── im2latexv2_test_normalized_results.txt
│   ├── im2latexv2_test_pred.json
│   └── im2latexv2_test_results.txt
└── test
    ├── images/
    └── test.json
```

## Evaluation Results (Normalized)

```text
============================================================
                    EVALUATION RESULTS
============================================================

Dataset Statistics:
  Total Samples: 10,117

Core Metrics:
  Mean Edit Score:        99.3159%
  BLEU-4 Score:           98.7041%
  Character Error Rate:    0.0126

Error Threshold Analysis:
  Exact Match Rate:       88.0300% (0 errors)
  Error ≤ 1:              92.9228%
  Error ≤ 2:              95.8189%
  Error ≤ 3:              97.0940%

Detailed Error Distribution:
  Errors ≤ 0:  8,906 samples (88.03%)
  Errors ≤ 1:  9,401 samples (92.92%)
  Errors ≤ 2:  9,694 samples (95.82%)
  Errors ≤ 3:  9,823 samples (97.09%)
```

---

# Usage Instructions

1. **Dataset Acquisition**:

   * Download images and prompts from respective dataset directories.

2. **Model Training and Testing**:

   * Use provided JSON prompts and images to train or evaluate models.

3. **Evaluation and Results**:

   * Analyze predictions in results directories (`*_pred.json`).
   * Review performance metrics in corresponding summary files (`*_results.txt`).

---

**Note**: Always verify file paths align with your local environment for effective use.
