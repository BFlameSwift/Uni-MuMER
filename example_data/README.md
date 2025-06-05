

# Example Data Folder

This folder provides supplementary materials supporting reproducibility and transparent evaluation of the results presented in the paper.

## Folder Structure

```
example_data
├── backup/
│   ├── crohme_2014.json
│   ├── crohme_2016.json
│   ├── crohme_2019.json
│   ├── hme100k_test.json
│   ├── im2latexv2_test.json
│   ├── mathwriting_test.json
│   ├── N1.json
│   ├── N2.json
│   └── N3.json
└── final_paper_results/
    ├── crohme_2014_results.txt
    ├── crohme_2016_results.txt
    ├── crohme_2019_results.txt
    ├── hme100k_test_results.txt
    ├── im2latexv2_test_results.txt
    ├── mathwriting_test_results.txt
    ├── N1_results.txt
    ├── N2_results.txt
    └── N3_results.txt
```

## Folder Descriptions

* **backup/**:
  Contains exact copies of the dataset files used for generating the results reported in the paper. These backups ensure reproducibility of experiments by providing consistent reference data.

* **final\_paper\_results/**:
  Contains the final prediction and evaluation results as presented in the published paper. These files are provided separately due to minor variability caused by the inference environment (e.g., top-k sampling in vLLM and hardware/environment differences).

## Important Note on Reproducibility

Due to the stochastic nature of inference (especially when using sampling methods such as top-k sampling in vLLM), minor differences in prediction outcomes are expected. The results in this directory are the official outcomes reported in the paper for transparent and verifiable research.

---
