# CVIB
This repo contains the PyTorch implementation for the paper ***Contrastive Variational Information Bottleneck for Aspect-based Sentiment Analysis*** (KBS 2023).

## Requirements
- Python 3.9.7
- PyTorch 1.11.0
- [Transformers](https://github.com/huggingface/transformers) 4.18.0
- CUDA 11.0


## Preparation
-  **BERT** <br>
Download the pytorch version pre-trained `bert-base-uncased` model and vocabulary from the link provided by huggingface <https://github.com/huggingface/transformers>. Then you can change the parameter `--bert_model_dir` to the directory of the bert model.

-  **Dependency Parser** <br>
Download the biaffine-dependency-parser-ptb-2020.04.06.tar.gz from <https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz> as `model.tar` to the `./data/pretrained-models` directory. (necessary if you would like to preprocess the data.)

## Data Preprocess
The codes for data preprocessing are in `data_preprocess_raw.py` (for REST15, REST16) and `data_preprocess_xml.py` (for REST14, LAP14, MAMS and ARTS). We have already provided the preprocessed data (including [ARTS Test Sets](https://github.com/zhijing-jin/ARTS_TestSet)) in `./ABSA_RGAT/`. Also, we have provided the preprocessed data for ASGCN-BERT (as the backbone) in `./ABSA_Graph/`.

## Training
Run the commands: ` bash train_xxx.sh `.  (For example, run `bash train_res14.sh` to train the model with REST14 dataset.)
