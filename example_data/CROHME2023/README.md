
# CROHME2023 Dataset

## Overview

The CROHME2023 dataset provides a test set for evaluating handwritten mathematical expression recognition systems. 

This release includes the original CROHME2023 test set and an additional normalized version. The normalization process standardized the LaTeX style and removed numerous invalid or problematic items, providing a cleaner test set for evaluation.

---

## Dataset Structure

```text
CROHME2023
├── crohme2023_test_normalized_pred.json
├── crohme2023_test_normalized_results.txt
├── crohme2023_test_pred.json
├── crohme2023_test_results.txt
├── crohme2023_val_normalized_pred.json
├── crohme2023_val_normalized_results.txt
├── crohme2023_val_pred.json
└── crohme2023_val_results.txt├── crohme2023_test_normalized_pred.json
├── crohme2023_test_normalized_results.txt
├── crohme2023_test_pred.json
├── crohme2023_test_results.txt
├── crohme2023_val_normalized_pred.json
├── crohme2023_val_normalized_results.txt
├── crohme2023_val_pred.json
└── crohme2023_val_results.txt
```

---

## Directory Explanation

* **CROHME2023\_test**:

  * Contains the original images of handwritten mathematical expressions along with corresponding captions.

* **prompts**:

  * `crohme2023_test.json`: Prompts associated with the original test set, we only add curly braces for clarity (`a _ i` → `a _ { i }`).
  * `crohme2023_test_normalized.json`: Prompts corresponding to the normalized dataset.
  * `crohme2023_val.json`: Prompts for the original validation set, we only add curly braces for clarity (`a _ i` → `a _ { i }`).
  * `crohme2023_val_normalized.json`: Prompts for the normalized validation set
  Example snippet:

  ```json
  [
    {
      "images": [
        "./data/CROHME2023/CROHME2023_test/images/form_5_241_E1202.jpg"
      ],
      "messages": [
        {
          "from": "human",
          "value": "<image>I have an image of a handwritten mathematical expression. Please write out the expression of the formula in the image using LaTeX format."
        },
        {
          "from": "gpt",
          "value": "z \\rightarrow x"
        }
      ]
    }
  ]
  ```

* **results**:

  * Evaluation results for both original and normalized datasets are included.
  * JSON files (`*_pred.json`) contain individual predictions and ground truth data.
  * TXT files (`*_results.txt`) summarize the overall performance metrics.

  Example of normalized dataset evaluation (`crohme2023_test_normalized_results.txt`):

  ```
  ============================================================
                      EVALUATION RESULTS
  ============================================================

  Dataset Statistics:
    Total Samples: 1,652

  Core Metrics:
    Mean Edit Score:        93.1409%
    BLEU-4 Score:           90.4988%
    Character Error Rate:    0.0568

  Error Threshold Analysis:
    Exact Match Rate:       69.4915% (0 errors)
    Error ≤ 1:              79.0557%
    Error ≤ 2:              88.3777%
    Error ≤ 3:              93.1598%

  Detailed Error Distribution:
    Errors ≤ 0:  1,148 samples (69.49%)
    Errors ≤ 1:  1,306 samples (79.06%)
    Errors ≤ 2:  1,460 samples (88.38%)
    Errors ≤ 3:  1,539 samples (93.16%)
  ```

---

## Usage Instructions

1. **Dataset Acquisition**:

   * Download images and prompts from the `CROHME2023_test` and `prompts` directories.

2. **Model Training and Testing**:

   * Use provided prompts (`crohme2023_test*.json`) alongside images to train, validate, or test recognition models.

3. **Evaluation and Results**:

   * Check predictions in `results` directories (`*_pred.json`) for detailed analysis.
   * Review performance metrics in the summary files (`*_results.txt`).

---

**Note**: Always verify file paths and ensure consistency with your local setup for smooth operation.
