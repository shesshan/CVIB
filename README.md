# CVIB
This repo contains the PyTorch implementation for our paper:

**[Contrastive Variational Information Bottleneck for Aspect-based Sentiment Analysis](https://www.sciencedirect.com/science/article/pii/S095070512301050X/pdfft?md5=5f85969d1933e1db0abbdaabea2365cd&pid=1-s2.0-S095070512301050X-main.pdf)**

Mingshan Chang, Min Yang, Qingshan Jiang, Ruifeng Xu. *Knowledge-Based Systems, 2024: 111302.*

Please cite our paper and kindly give a star if you use this repo~

## Requirements
- Python 3.9.7
- PyTorch 1.11.0
- [Transformers](https://github.com/huggingface/transformers) 4.18.0
- CUDA 11.0


## Preparation
-  **BERT** <br>
Download the pytorch version pre-trained `bert-base-uncased` model and vocabulary from the link provided by huggingface <https://github.com/huggingface/transformers>. Then you can change the parameter `--bert_model_dir` to the directory of the bert model.

-  **Dependency Parser** <br>
Download [biaffine-dependency-parser-ptb-2020.04.06.tar.gz](https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz) to `./parser/`. (necessary if you'd like to preprocess the data.)

## Data Preprocess
The codes for data preprocessing are in `data_preprocess_raw.py` (for REST15, REST16) and `data_preprocess_xml.py` (for REST14, LAP14, MAMS and ARTS). We have already provided the preprocessed data (including [ARTS Test Sets](https://github.com/zhijing-jin/ARTS_TestSet)) in `./ABSA_RGAT/`. Also, we have provided the preprocessed data for ASGCN-BERT (as the backbone) in `./ABSA_Graph/`.

## Training
Run the commands: ` bash train_xxx.sh `.  (For example, run `bash train_res14.sh` to train the model with REST14 dataset.)

## Citation
The BibTex of the citation is as follow:
```
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
