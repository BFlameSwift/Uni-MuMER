

# HME100K Dataset

## Overview

The HME100K dataset provides a comprehensive test set for handwritten mathematical expressions. 


This release includes the original test set and an additional normalized version. The normalized set standardizes LaTeX formatting and excludes items containing CJK (Chinese, Japanese, Korean) characters, ensuring improved consistency and relevance.
## Dataset Structure

```text
HME100K
├── prompts
│   ├── hme100k_test.json
│   └── hme100k_test_normalized.json
├── results
│   ├── hme100k_test_normalized_pred.json
│   ├── hme100k_test_normalized_results.txt
│   ├── hme100k_test_pred.json
│   └── hme100k_test_results.txt
└── test
    ├── images/
    └── test_caption.txt
```

## Directory Explanation

* **test**:

  * Contains images of handwritten mathematical expressions and their corresponding captions.

* **prompts**:

  * `hme100k_test.json`: Original prompts.
  * `hme100k_test_normalized.json`: Prompts after normalization and removal of CJK items.

* **results**:

  * Includes predictions and performance evaluations.

  Example snippet (`hme100k_test_normalized_results.txt`):

  ```text
  ============================================================
                      EVALUATION RESULTS
  ============================================================

  Dataset Statistics:
    Total Samples: 24,016

  Core Metrics:
    Mean Edit Score:        96.8736%
    BLEU-4 Score:           95.1607%
    Character Error Rate:    0.0287

  Error Threshold Analysis:
    Exact Match Rate:       72.7224% (0 errors)
    Error ≤ 1:              87.6666%
    Error ≤ 2:              92.5758%
    Error ≤ 3:              94.8951%

  Detailed Error Distribution:
    Errors ≤ 0: 17,465 samples (72.72%)
    Errors ≤ 1: 21,054 samples (87.67%)
    Errors ≤ 2: 22,233 samples (92.58%)
    Errors ≤ 3: 22,790 samples (94.90%)
  ```

---

# Usage Instructions

1. **Dataset Acquisition**:

   * Download images and prompts from the respective `test` and `prompts` directories.

2. **Model Training and Testing**:

   * Utilize provided prompts (`*_test*.json`) alongside images for training or evaluation.

3. **Evaluation and Results**:

   * Inspect predictions in the `results` directories (`*_pred.json`).
   * Review summary performance metrics in the results files (`*_results.txt`).

---

**Note**: Ensure all file paths align with your local setup for seamless integration.
