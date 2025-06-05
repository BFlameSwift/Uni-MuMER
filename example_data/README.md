

# Example Data Folder

This folder provides supplementary materials supporting reproducibility, transparent evaluation, and verification of results presented in our paper.

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
├── final_paper_results/
│   ├── crohme_2014_results.txt
│   ├── crohme_2016_results.txt
│   ├── crohme_2019_results.txt
│   ├── hme100k_test_results.txt
│   ├── im2latexv2_test_results.txt
│   ├── mathwriting_test_results.txt
│   ├── N1_results.txt
│   ├── N2_results.txt
│   └── N3_results.txt
├── CROHME/
├── CROHME2023/
├── HME100K/
├── Im2LaTeXv2/
├── MathWriting/
└── MNE/
```

## Folder Descriptions

### Core Data Directories

* **backup/**
  Contains exact copies of dataset files used for generating the results reported in the paper. These backups ensure the reproducibility of experiments by providing consistent and stable reference data.

* **final\_paper\_results/**
  Contains the official evaluation results as presented in the published paper. These results may differ slightly from reproduced runs due to environmental variations, including inference methods (e.g., top-k sampling in vLLM) and computational hardware differences.

### Dataset Directories

The following directories provide detailed dataset structures and related evaluation resources. For full datasets, please refer to the [Google Drive link](https://drive.google.com/drive/folders/1T8a3WxICZVl1NJ99hu9tuuqqNZoxGhXq?usp=sharing).

* **CROHME/**
* **CROHME2023/**
* **HME100K/**
* **Im2LaTeXv2/**
* **MathWriting/**
* **MNE/**

Each dataset directory contains specific prompts, images, and result files structured similarly for ease of use.



## Important Note on Reproducibility

Due to the inherent stochastic nature of inference processes—particularly when employing sampling-based methods such as vLLM's top-k sampling—minor discrepancies in reproduced prediction results are expected. The `final_paper_results/` directory provides authoritative results officially reported in the publication, ensuring transparent and verifiable research practices.


