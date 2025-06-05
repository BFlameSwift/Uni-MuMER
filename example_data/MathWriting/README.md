

# MathWriting Dataset

## Overview

The MathWriting dataset provides test sets designed to assess recognition systems specifically on handwritten mathematical expressions. The dataset includes original and normalized versions to ensure consistency and improved alignment with training conventions.

## Dataset Structure

```text
MathWriting
├── prompts
│   ├── mathwriting_test.json
│   └── mathwriting_test_normalized.json
├── results
│   ├── mathwriting_test_normalized_pred.json
│   ├── mathwriting_test_normalized_results.txt
│   ├── mathwriting_test_pred.json
│   └── mathwriting_test_results.txt
└── test
    ├── images/
    └── test_items.json
```

## Evaluation Results (Original)

```text
============================================================
                    EVALUATION RESULTS
============================================================

Dataset Statistics:
  Total Samples: 7,644

Core Metrics:
  Mean Edit Score:        90.7554%
  BLEU-4 Score:           84.8671%
  Character Error Rate:    0.0921

Error Threshold Analysis:
  Exact Match Rate:       51.4129% (0 errors)
  Error ≤ 1:              67.5170%
  Error ≤ 2:              76.7661%
  Error ≤ 3:              83.9744%

Detailed Error Distribution:
  Errors ≤ 0:  3,930 samples (51.41%)
  Errors ≤ 1:  5,161 samples (67.52%)
  Errors ≤ 2:  5,868 samples (76.77%)
  Errors ≤ 3:  6,419 samples (83.97%)
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
