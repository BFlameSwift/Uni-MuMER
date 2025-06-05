

# MNE Dataset

## Overview

The MNE dataset provides test sets specifically curated for evaluating handwritten mathematical expressions. The dataset is divided into three subsets (N1, N2, N3), each containing images, captions, and (for N1 and N2) dictionaries for reference.

## Dataset Structure

```text
MNE
├── N1
│   ├── caption.txt
│   ├── dictionary.txt
│   └── images/
├── N2
│   ├── caption.txt
│   ├── dictionary.txt
│   └── images/
├── N3
│   ├── caption.txt
│   └── images/
├── prompts
│   ├── N1.json
│   ├── N2.json
│   └── N3.json
└── results
    ├── N1_pred.json
    ├── N1_results.txt
    ├── N2_pred.json
    ├── N2_results.txt
    ├── N3_pred.json
    └── N3_results.txt
```

## Evaluation Results (N1 Example)

```text
============================================================
                    EVALUATION RESULTS
============================================================

Dataset Statistics:
  Total Samples: 1,875

Core Metrics:
  Mean Edit Score:        96.7094%
  BLEU-4 Score:           92.2334%
  Character Error Rate:    0.0249

Error Threshold Analysis:
  Exact Match Rate:       76.1600% (0 errors)
  Error ≤ 1:              88.0000%
  Error ≤ 2:              92.7467%
  Error ≤ 3:              95.3067%

Detailed Error Distribution:
  Errors ≤ 0:  1,428 samples (76.16%)
  Errors ≤ 1:  1,650 samples (88.00%)
  Errors ≤ 2:  1,739 samples (92.75%)
  Errors ≤ 3:  1,787 samples (95.31%)
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
