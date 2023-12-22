# coding=utf-8
import sys
from trainer import train_vib
from model import Aspect_Bert_GAT, Aspect_Bert_GCN
from custom_datasets import load_datasets_and_vocabs, ABSADataset_from_Raw
from transformers import BertTokenizer, BertConfig
import torch
import numpy as np
import random
import argparse
import logging
import os
from transformers import AdamW
from torch import optim
# default, if not specify CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

OPTIMIZERS = {
    'adam': torch.optim.Adam,  # default: lr=1e-3, eps=1e-8, weight_decay=0.0
    'adamw': AdamW,  # default: lr=1e-3, eps=1e-6, weight_decay=0.0
}

MODELS = {
    'asgcn_bert': Aspect_Bert_GCN,
    'rgat_bert': Aspect_Bert_GAT
}


def setup_logger(filepath):
    global logger
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    _format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(_format)
    ch.setFormatter(_format)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        '--source_domain',
        type=str,
        default='rest',
        choices=['rest14', 'lap14', 'rest15', 'rest16', 'mams'],
        help='ABSA dataset for training.')
    parser.add_argument(
        '--target_domain',
        type=str,
        default='rest',
        choices=['rest14', 'lap14', 'rest15', 'rest16', 'mams'],
        help='ABSA dataset for testing.')
    # Robustness & Generalization Test
    parser.add_argument('--arts_test',
                        action='store_true',
                        help='Robustness Test on ARTS test set.')
    parser.add_argument('--cross_domain',
                        action='store_true',
                        help='Out-Of-Domain Generalization Test.')
    parser.add_argument('--case_study',
                        action='store_true',
                        help='Case study.')
    parser.add_argument(
        '--data_root_dir',
        type=str,
        default='/data0/mschang/ABSA_RGAT/',
        help=
        'Directory to store ABSA data, such as raw data, vocab, embeddings, tags_vocab, etc.'
    )
    parser.add_argument('--num_classes',
                        type=int,
                        default=3,
                        help='Number of classes of ABSA.')
    parser.add_argument('--save_folder',
                        type=str,
                        default='./results',
                        help='Directory to store trained model.')
    parser.add_argument('--log_file',
                        default='run.log',
                        type=str,
                        help='location of log file')
    parser.add_argument(
        '--config_file',
        default='./config/bert_config.json',
        type=str,
        help='location of BERT custom config file if specified.')
    parser.add_argument('--cuda_id',
                        type=str,
                        default='3',
                        help='Choose which GPUs to run')
    parser.add_argument('--seed',
                        type=int,
                        default=2019,
                        help='random seed for initialization')

    # Model parameters
    parser.add_argument('--embedding_type',
                        type=str,
                        default='bert',
                        choices=['glove', 'bert'])
    parser.add_argument('--glove_dir',
                        type=str,
                        default='/data0/mschang/glove',
                        help='Directory of GloVe embeddings.')
    parser.add_argument('--embedding_dim',
                        type=int,
                        default=300,
                        help='GloVe embeddings dimension.')
    parser.add_argument('--bert_model_dir',
                        type=str,
                        default='bert-base-uncased',
                        help='Path to pre-trained Bert model.')
    # 10.29 add
    parser.add_argument(
        '--model_name',
        type=str,
        default='rgat_bert',
        choices=['asgcn_bert', 'rgat_bert'],
        help='backbone model.')

    parser.add_argument('--spc',
                        action='store_true',
                        default=False,
                        help='use sentence-aspect pair as input.')

    parser.add_argument('--highway',
                        action='store_true',
                        help='Use highway embed.')
    parser.add_argument('--num_layers',
                        type=int,
                        default=2,
                        help='Number of layers of BiLSTM or Highway or Elmo.')
    parser.add_argument(
        '--add_non_connect',
        type=bool,
        default=True,
        help=
        'Add a sepcial "non-connect" relation for aspect with no direct connection.'
    )
    parser.add_argument('--multi_hop',
                        type=bool,
                        default=True,
                        help='Multi hop non connection.')
    parser.add_argument('--max_hop',
                        type=int,
                        default=4,
                        help='max number of hops')
    parser.add_argument('--num_heads',
                        type=int,
                        default=6,
                        help='Number of attention heads of GAT.')
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout rate for embedding & representations before MLP layers.')
    # ASGCN [10.28 add]
    parser.add_argument('--num_gcn_layers',
                        type=int,
                        default=1,
                        help='Number of GCN layers.')
    parser.add_argument('--gcn_dropout',
                        type=float,
                        default=0.3,
                        help='Dropout rate for GCN.')
    # R-GAT
    parser.add_argument('--gat_our', action='store_true', help='R-GAT')
    parser.add_argument('--gat_attention_type',
                        type=str,
                        choices=['linear', 'dotprod', 'gcn'],
                        default='dotprod',
                        help='The attention used for gat')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=300,
                        help='Hidden size of BiLSTM, in early stage.')
    parser.add_argument(
        '--final_hidden_size',
        type=int,
        default=300,
        help=
        'Hidden size of MLP layers i.e. dim of final output representations.')
    parser.add_argument('--num_mlps',
                        type=int,
                        default=2,
                        help='Number of MLP layers.')
    parser.add_argument('--gat_dropout',
                        type=float,
                        default=0,
                        help='Dropout rate for GAT.')
    # CLR
    parser.add_argument('--prune',
                        action='store_true',
                        default=False,
                        help='self pruning')
    parser.add_argument('--sd_temp', default=0.1, type=float)
    parser.add_argument('--sdcl_fac',
                        type=float,
                        default=1.0,
                        help="alpha factor of self-damaging contrastive loss")
    parser.add_argument('--prune_percent',
                        type=float,
                        default=0.3,
                        help="whole prune percentage")
    parser.add_argument('--random_prune_percent',
                        type=float,
                        default=0.0,
                        help="random prune percentage")

    # VIB
    parser.add_argument('--use_ib',
                        action='store_true',
                        default=False,
                        help='using VIB.')
    parser.add_argument('--kl_fac',
                        type=float,
                        default=1e-6,
                        help="KL loss term factor")
    parser.add_argument(
        '--ib_lr',
        type=float,
        default=-1,
        help=
        'Separate learning rate for information bottleneck params. Set to -1 to follow args.lr.'
    )
    parser.add_argument(
        '--ib_wd',
        type=float,
        default=-1,
        help=
        'Separate weight decay for information bottleneck params. Set to -1 to follow args.weight_decay'
    )
    parser.add_argument('--lr_fac',
                        type=float,
                        default=0.5,
                        help='LR decreasing factor.')
    parser.add_argument('--lr_epoch',
                        type=int,
                        default=10,
                        help='Decrease (vib) learning rate every x epochs.')
    parser.add_argument(
        '--first_3layers',
        action='store_true',
        default=False,
        help='add VIB-based masking layer into first 3 BERT layers')
    parser.add_argument(
        '--last_3layers',
        action='store_true',
        default=False,
        help='add VIB-based masking layer into last 3 BERT layers')

    # Training parameters
    parser.add_argument("--per_gpu_train_batch_size",
                        default=16,
                        type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size",
                        default=32,
                        type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=2,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument('--max_seq_len',
                        type=int,
                        default=128,
                        help='max length of input sequences.')
    parser.add_argument("--optimizer_type",
                        default='adamw',
                        type=str,
                        help="Type of optimizer, adam or adamw.")
    parser.add_argument("--lr",
                        default=1e-3,
                        type=float,
                        help="Initial learning rate.")
    parser.add_argument("--nd_lr",
                        default=1e-3,
                        type=float,
                        help="Initial learning rate of no-decay params.")
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="Weight decay.")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for optimizer.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs",
                        default=30.0,
                        type=float,
                        help="Total number of training epochs.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help=
        "If > 0: set total number of training steps(that update the weights) to perform. Override num_train_epochs."
    )
    parser.add_argument('--logging_steps',
                        type=int,
                        default=50,
                        help="Log every X updates steps.")

    return parser.parse_args()


def check_args(args):
    '''
    eliminate params redundancy
    '''
    logging.info(vars(args))


def write_to_csv(save_path, results):
    import csv
    with open(save_path, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(results)
    f.close()


def main():

    # Parse args
    args = parse_args()
    check_args(args)

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    save_folder = '{}/{}'.format(args.save_folder, args.model_name)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    save_folder += '/{}-{}'.format(
        args.source_domain,
        args.target_domain + '-ARTS' if args.arts_test else args.target_domain)

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    args.save_folder = save_folder

    # Logging file
    if args.log_file:
        log_dir = '{}/logs'.format(args.save_folder)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        log_file = '{}/{}'.format(log_dir, args.log_file)
        setup_logger(log_file)

    # Setup GPU training
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    logging.info('Training Device: %s', str(args.device).upper())

    # Set seed
    set_seed(args)

    # Bert, load pretrained model and tokenizer, check if neccesary to put bert here
    if args.embedding_type == 'bert':
        # embedding_type: glove OR bert
        tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir)

    # Load datasets and vocabs
    if 'gcn' in args.model_name:
        train_dataset = ABSADataset_from_Raw(args)
        test_dataset = ABSADataset_from_Raw(args, do_train=False)
    else:
        train_dataset, test_dataset, word_vocab, dep_tag_vocab, pos_tag_vocab = load_datasets_and_vocabs(
            args)

    # config_file can be customized json file or pretrain-model load path
    if args.config_file:
        config = BertConfig.from_pretrained(args.config_file)
        logging.info('Load customized config file for BERT.')
    else:
        config = BertConfig.from_pretrained(args.bert_model_dir)

    if args.first_3layers:
        logging.info(
            'Add VIB-based masking layer into the first 3 BERT layers.')
        config.first3 = True

    if args.last_3layers:
        logging.info(
            'Add VIB-based masking layer into the last 3 BERT layers.')
        config.last3 = True

    config.use_ib = args.use_ib

    # Build Model
    model = MODELS[args.model_name]
    main_model, sub_model = None, None
    if 'rgat' in args.model_name:
        # RGAT-BERT
        sub_model = model(args, config, dep_tag_vocab['len'],
                          main_tag=False)  # self-pruned network
        main_model = model(args, config, dep_tag_vocab['len'],
                           main_tag=True)  # original network
    else: 
        # other backbone
        sub_model = model(args, config, main_tag=False)
        main_model = model(args, config, main_tag=True)

    main_model.to(args.device)
    sub_model.to(args.device)

    args.optimizer = OPTIMIZERS[args.optimizer_type]

    main_eval_results, sub_eval_results = [], []
    genenral_params = [
        args.num_train_epochs,
        args.max_seq_len if 'gcn' in args.model_name else 'batch_first',
        args.per_gpu_train_batch_size, args.per_gpu_eval_batch_size,
        args.optimizer_type, args.lr, args.nd_lr, args.weight_decay,
        args.max_grad_norm, args.num_mlps, args.final_hidden_size, args.dropout
    ]
    if 'gcn' in args.model_name:
        genenral_params.append(args.gcn_dropout)
    elif 'gat' in args.model_name:
        genenral_params.append(args.gat_dropout)
    # Train
    main_eval_results, sub_eval_results = train_vib(args, main_model,
                                                    sub_model, train_dataset,
                                                    test_dataset)
    params_and_results = genenral_params + [
        args.kl_fac, args.ib_lr, args.ib_wd, args.sdcl_fac, args.sd_temp
    ]

    if len(main_eval_results):
        best_eval_acc = max(main_eval_results, key=lambda x: x['acc'])
        best_eval_f1 = max(main_eval_results, key=lambda x: x['f1'])
        for key in sorted(best_eval_acc.keys()):
            logging.info('[Max Accuracy of Main Network]')
            logging.info(' {} = {}'.format(key, str(best_eval_acc[key])))
            params_and_results.append(best_eval_acc[key])
        for key in sorted(best_eval_f1.keys()):
            logging.info('[Max F1 of Main Network]')
            logging.info(' {} = {}'.format(key, str(best_eval_f1[key])))

    if len(sub_eval_results):
        best_eval_acc = max(sub_eval_results, key=lambda x: x['acc'])
        best_eval_f1 = max(sub_eval_results, key=lambda x: x['f1'])
        for key in sorted(best_eval_acc.keys()):
            logging.info('[Max Accuracy of Sub Network]')
            logging.info(' {} = {}'.format(key, str(best_eval_acc[key])))
            params_and_results.append(best_eval_acc[key])
        for key in sorted(best_eval_f1.keys()):
            logging.info('[Max F1 of Sub Network]')
            logging.info(' {} = {}'.format(key, str(best_eval_f1[key])))

    write_to_csv(save_path='{}/results_{}.csv'.format(
        args.save_folder, args.model_name if args.use_ib else 'pure'),
                 results=params_and_results)


if __name__ == "__main__":
    main()
