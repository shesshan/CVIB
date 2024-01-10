![](https://img.shields.io/badge/version-1.0.1-blue)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/shesshan/CVIB/blob/main/LICENSE)
[![AAAI-24](https://img.shields.io/badge/KBS-2024-black?color=%232E8B57)]()
[![Pytorch](https://img.shields.io/badge/PyTorch-%23FF6347.svg?e&logo=PyTorch&logoColor=white)](https://pytorch.org/)

# 🪄CVIB
Hi there👋, this repo contains the PyTorch implementation for our paper:

**[Contrastive Variational Information Bottleneck for Aspect-based Sentiment Analysis](https://www.sciencedirect.com/science/article/pii/S095070512301050X/pdfft?md5=5f85969d1933e1db0abbdaabea2365cd&pid=1-s2.0-S095070512301050X-main.pdf)**

Mingshan Chang, Min Yang, Qingshan Jiang, Ruifeng Xu. *Knowledge-Based Systems, 2024: 111302.*

## 📜 Summary
> 🔎 Despite the effectiveness, deep ABSA models are susceptible to 🫧***spurious correlations***🫧 between input features and output labels, which in general suffer from poor robustness and generalization.

For better understanding, we provide an example of the spurious correlation problem in ABSA:

<img src="/docs/example_00.jpg" width = "50%" />

To address this challenge, we propose a novel **C**ontrastive **V**ariational **I**nformation **B**ottleneck framework (called **CVIB**), encompassing an original network and a self-pruned network. These two networks are optimized simultaneously via contrastive learning.
- We employ the variational information bottleneck (VIB) principle to learn an informative and compressed network (self-pruned network) from the original network, which discards the spurious correlations while preserving sufficient information about the sentiment labels.
- A self-pruning contrastive loss is devised to optimize these two networks, where the representations learned by two networks are regarded as a semantically similar positive pair while representations of two different instances within a mini-batch are treated as a negative pair. 

## 🧩 Architecture
<img src="/docs/cvib_frm_revise_v4_00.jpg" width = "85%" />
CVIB is composed of an original network and a self-pruned network, where the self-pruned network is learned adaptively from the original network based on the VIB principle. The self-pruned network is expected to discard the spurious correlations between input features and output prediction, which is used for inference.

## 🎯 Main Results
<img src="/docs/main_results.png" width = "90%" />

## 🗂 Code & Data

### Requirements
- Python 3.9.7
- PyTorch 1.11.0
- [Transformers](https://github.com/huggingface/transformers) 4.18.0
- CUDA 11.0

### Preparation
-  **BERT** <br>
Download the pytorch version [bert-base-uncased](https://github.com/huggingface/transformers) from huggingface. Then, you can set the parameter `--bert_model_dir` to the local directory of the BERT model.

-  **Dependency Parser** <br>
Download [biaffine-dependency-parser-ptb-2020.04.06.tar.gz](https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz) to `./parser/` to build the dependency tree for review sentences. (necessary if you'd like to preprocess the data yourself.)

### Data Pre-process
- Code for data preprocessing can be found in [data_preprocess_raw.py](/data_preprocess_raw.py) (for REST15, REST16) and [data_preprocess_xml.py](/data_preprocess_xml.py) (for REST14, LAP14, MAMS and [ARTS](https://github.com/zhijing-jin/ARTS_TestSet)). 

- The preprocessed data can be found in [ABSA_RGAT/](/ABSA_RGAT/). Also, we have provided the preprocessed data in [ABSA_Graph/](/ABSA_Graph/) (for choosing ASGCN-BERT as the backbone).

### Training
- Run the commands: ` bash train_xxx.sh `, e.g. run `bash train_res14.sh` to train with REST14 dataset.

## Citation
The BibTex of the citation is as follow:
```bibtex
@article{CHANG2024111302,
title = {Contrastive variational information bottleneck for aspect-based sentiment analysis},
author = {Mingshan Chang and Min Yang and Qingshan Jiang and Ruifeng Xu},
journal = {Knowledge-Based Systems},
volume = {284},
pages = {111302},
year = {2024},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2023.111302},
url = {https://www.sciencedirect.com/science/article/pii/S095070512301050X}
}
```

🤘Please cite our paper and kindly give a star if you find this repo useful.💡
