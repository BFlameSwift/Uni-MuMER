# CROHME Dataset

## Overview

The CROHME (Competition on Recognition of Handwritten Mathematical Expressions) dataset is a benchmark for evaluating recognition systems of handwritten mathematical expressions. It includes images of handwritten formulas along with their corresponding ground truth annotations.

---

## Dataset Structure

These datasets is structured as follows:
```
CROHME
├── 2014
│   ├── caption.txt
│   └── images/
├── 2016
│   ├── caption.txt
│   └── images/
├── 2019
│   ├── caption.txt
│   └── images/
├── prompts
│   ├── crohme_2014.json
│   ├── crohme_2016.json
│   └── crohme_2019.json
├── results
│   ├── crohme_2014_pred.json
│   ├── crohme_2014_results.txt
│   ├── crohme_2016_pred.json
│   ├── crohme_2016_results.txt
│   ├── crohme_2019_pred.json
│   └── crohme_2019_results.txt
└── README.md
```


### Directory Explanation
- 2014, 2016, 2019: Each year-specific directory contains:
- images/: Image files of handwritten mathematical expressions.
- caption.txt: Captions or descriptions related to the images.
- prompts: Contains JSON files (crohme_\<year>.json) that define prompts for using the images in training or inference tasks. Each prompt includes a reference image and the expected LaTeX representation.

    Example snippet:
    ```json
    [
        {
            "images": [
               "./data/CROHME/2014/images/514_em_341.jpg"
            ],
            "messages": [
                {
                    "from": "human",
                    "value": "<image>I have an image of a handwritten mathematical expression. Please write out the expression of the formula in the image using LaTeX format."
                },
                {
                    "from": "gpt",
                    "value": "\\tan \\alpha _ { i }"
                }
            ]
        }
    ]
    ```

- results: Contains evaluation outcomes and predictions from experiments.
- crohme_<year>_pred.json: JSON files listing each image’s ground truth (gt), model predictions (pred), image path (image_path), and image ID (img_id).
- crohme_<year>_results.txt: Plain-text files summarizing evaluation metrics such as Mean Edit Score, BLEU-4 Score, and Character Error Rate.
Example snippet (crohme_2014_results.txt):



### EVALUATION RESULTS

```text
Dataset Statistics:
  Total Samples: 986

Core Metrics:

    Mean Edit Score:        96.5522%
    BLEU-4 Score:           89.1164%
    Character Error Rate:    0.0396

Error Threshold Analysis:
    Exact Match Rate:       82.6572% (0 errors)
    Error ≤ 1:              90.3651%
    Error ≤ 2:              93.7120%
    Error ≤ 3:              96.4503%

Detailed Error Distribution:
    Errors ≤ 0:    815 samples (82.66%)
    Errors ≤ 1:    891 samples (90.37%)
    Errors ≤ 2:    924 samples (93.71%)
    Errors ≤ 3:    951 samples (96.45%)
```




Note: Ensure all file paths referenced in JSON files correspond to your local directory structure for seamless integration.