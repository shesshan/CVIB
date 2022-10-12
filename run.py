# coding=utf-8
import sys
from trainer import train, train_single_vib, train_vib, train_sdclr
from model import *
from custom_datasets import load_datasets_and_vocabs
from transformers import BertTokenizer, BertConfig
import torch
import numpy as np
import random
import argparse
import logging
import os
# default, if not specify CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


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
    parser.add_argument('--source_domain', type=str, default='rest',
                        choices=['rest', 'laptop', 'twitter',
                                 'rest15', 'rest16', 'mams'],
                        help='Choose ABSA dataset of source domain for training.')
    parser.add_argument('--target_domain', type=str, default='rest',
                        choices=['rest', 'laptop', 'twitter',
                                 'rest15', 'rest16', 'mams'],
                        help='Choose ABSA dataset of target domain for testing.')
    # Robustness & Generalization Test
    parser.add_argument('--arts_test', action='store_true',
                        help='Robustness Test on ARTS test set.')
    parser.add_argument('--cross_domain', action='store_true',
                        help='Out-Of-Domain Generalization Test.')
    parser.add_argument('--case_study', action='store_true',
                        help='Case study.')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to store intermedia data, such as vocab, embeddings, tags_vocab.')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of classes of ABSA.')
    parser.add_argument('--save_folder', type=str, default='./results',
                        help='Directory to store trained model.')

    parser.add_argument('--log_file', default='run.log',
                        type=str, help='location of log file')
    parser.add_argument('--config_file', default='bert_config.json',
                        type=str, help='location of BERT custom config file if specified.')

    parser.add_argument('--cuda_id', type=str, default='3',
                        help='Choose which GPUs to run')
    parser.add_argument('--seed', type=int, default=2019,
                        help='random seed for initialization')

    # Model parameters
    parser.add_argument('--embedding_type', type=str,
                        default='glove', choices=['glove', 'bert'])
    parser.add_argument('--embedding_dim', type=int, default=300,
                        help='Dimension of glove embeddings')
    parser.add_argument('--glove_dir', type=str, default='/data1/mschang/glove',
                        help='Directory storing glove embeddings')
    parser.add_argument('--bert_model_dir', type=str, default='bert-base-uncased',
                        help='Path to pre-trained Bert model.')
    parser.add_argument('--pure_bert', action='store_true',
                        help='use BERT-SPC model')
    parser.add_argument('--capsnet_bert', action='store_true',
                        help='use CapsNet-BERT model')
    parser.add_argument('--gat_bert', action='store_true',
                        help='use RGAT-BERT model.')

    parser.add_argument('--spc', action='store_true', default=False,
                        help='use sentence-aspect pair as input.')

    parser.add_argument('--highway', action='store_true',
                        help='Use highway embed.')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers of bilstm or highway or elmo.')
    parser.add_argument('--add_non_connect',  type=bool, default=True,
                        help='Add a sepcial "non-connect" relation for aspect with no direct connection.')
    parser.add_argument('--multi_hop',  type=bool, default=True,
                        help='Multi hop non connection.')
    parser.add_argument('--max_hop', type=int, default=4,
                        help='max number of hops')

    parser.add_argument('--num_heads', type=int, default=6,
                        help='Number of heads for gat.')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout rate for embedding.')

    parser.add_argument('--num_gcn_layers', type=int, default=1,
                        help='Number of GCN layers.')
    parser.add_argument('--gcn_mem_dim', type=int, default=300,
                        help='Dimension of the W in GCN.')
    parser.add_argument('--gcn_dropout', type=float, default=0.2,
                        help='Dropout rate for GCN.')
    # R-GAT & ASGCN
    parser.add_argument('--gat_our', action='store_true',
                        help='R-GAT')
    parser.add_argument('--gat_attention_type', type=str, choices=['linear', 'dotprod', 'gcn'], default='dotprod',
                        help='The attention used for gat')
    parser.add_argument('--dep_relation_embed_dim', type=int, default=300,
                        help='Dimension for dependency relation embeddings.')
    parser.add_argument('--hidden_size', type=int, default=300,
                        help='Hidden size of bilstm, in early stage.')
    parser.add_argument('--final_hidden_size', type=int, default=300,
                        help='Hidden size of MLP layers, in early stage.')
    parser.add_argument('--num_mlps', type=int, default=2,
                        help='Number of mlps in the last of model.')
    # ATAE-LSTM
    parser.add_argument('--atae_lstm', action='store_true',
                        help='ATAE-LSTM')
    # MemNet
    parser.add_argument('--memnet', action='store_true',
                        help='MemNet')
    parser.add_argument('--hops', type=int, default=7,
                        help='number of hops in deep memory network, 7 for RES14 and 9 for LAP14')
    # IAN
    parser.add_argument('--ian', action='store_true',
                        help='IAN')
    # CLR
    parser.add_argument('--prune', action='store_true',
                        default=False, help='self pruning')
    parser.add_argument('--sd_temp', default=0.1, type=float)
    parser.add_argument('--sdcl_fac', type=float,
                        default=1.0, help="alpha factor of self-damaging contrastive loss")
    parser.add_argument('--prune_percent', type=float,
                        default=0.3, help="whole prune percentage")
    parser.add_argument('--random_prune_percent', type=float,
                        default=0.0, help="random prune percentage")

    # VIB
    parser.add_argument('--kl_fac', type=float,
                        default=1e-6, help="KL loss term factor")
    parser.add_argument('--ib_lr', type=float, default=-1,
                        help='Separate learning rate for information bottleneck params. Set to -1 to follow args.lr.')
    parser.add_argument('--ib_wd', type=float, default=-1,
                        help='Separate weight decay for information bottleneck params. Set to -1 to follow args.weight_decay')
    parser.add_argument('--lr_fac', type=float, default=0.5,
                        help='LR decreasing factor.')
    parser.add_argument('--lr_epoch', type=int, default=10,
                        help='Decrease (vib) learning rate every x epochs.')
    parser.add_argument('--first_3layers', action='store_true',
                        default=False, help='add VIB-based masking layer into first 3 BERT layers')
    parser.add_argument('--last_3layers', action='store_true',
                        default=False, help='add VIB-based masking layer into last 3 BERT layers')

    # Training parameters
    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--lr", default=1e-3, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--nd_lr", default=1e-3, type=float,
                        help="The initial learning rate of no-weight-decay params.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps(that update the weights) to perform. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")

    return parser.parse_args()


def check_args(args):
    '''
    eliminate confilct situations

    '''
    logging.info(vars(args))


def main():

    # Parse args
    args = parse_args()
    check_args(args)

    if args.pure_bert:
        save_folder = str(args.save_folder)+'/bert'
    elif args.capsnet_bert:
        save_folder = str(args.save_folder)+'/caps_bert'
    elif args.gat_bert:
        save_folder = str(args.save_folder)+'/rgat_bert'
    elif args.gat_our:
        if str(args.gat_attention_type) == 'gcn':
            save_folder = str(args.save_folder)+'/asgcn'
        else:
            save_folder = str(args.save_folder)+'/rgat'
    elif args.atae_lstm:
        save_folder = str(args.save_folder)+'/atae_lstm'
    elif args.memnet:
        save_folder = str(args.save_folder)+'/memnet'
    elif args.ian:
        save_folder = str(args.save_folder)+'/ian'

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    save_folder += '/' + \
        str(args.source_domain)+'-'+str(args.target_domain)

    if args.arts_test:
        save_folder += '-ARTS'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    args.save_folder = save_folder

    if args.log_file:
        log_dir = str(args.save_folder) + '/logs'
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        log_file = log_dir + '/' + args.log_file
        setup_logger(log_file)

    # Setup CUDA, GPU training
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    logging.info('Device is %s', str(args.device).upper())

    # Set seed
    set_seed(args)

    # Bert, load pretrained model and tokenizer, check if neccesary to put bert here
    if args.embedding_type == 'bert':  # embedding_type: glove OR bert
        tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir)
        args.tokenizer = tokenizer

    # Load datasets and vocabs
    # [6.7 add] RGAT's own datasets loading method.
    train_dataset, test_dataset, word_vocab, dep_tag_vocab, pos_tag_vocab = load_datasets_and_vocabs(
        args)

    # config_file can be customized json file or pretrain-model load path
    if args.config_file:
        config = BertConfig.from_pretrained(args.config_file)
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

    if config.use_ib:
        main_tag = False  # VIB
    else:
        main_tag = True  # no-VIB

    sub_model = None
    # Build Model
    if args.gat_our:
        main_model = Aspect_Relational_GAT(
            args=args, dep_tag_num=dep_tag_vocab['len'])
    elif args.atae_lstm:
        main_model = ATAE_LSTM(args)
    elif args.memnet:
        main_model = MemNet(args)
    elif args.ian:
        main_model = IAN(args)
    elif args.pure_bert:
        if args.prune:
            sub_model = Pure_Bert(args, config, main_tag=main_tag)
            main_tag = True
        main_model = Pure_Bert(args, config, main_tag=main_tag)
    elif args.capsnet_bert:
        if args.prune:
            sub_model = CapsNet_Bert(args, config, main_tag=main_tag)
            main_tag = True
        main_model = CapsNet_Bert(args, config, main_tag=main_tag)
    elif args.gat_bert:
        if args.prune:
            sub_model = Aspect_Bert_GAT(
                args, config, dep_tag_vocab['len'], main_tag=main_tag)  # pruned network
            main_tag = True
        main_model = Aspect_Bert_GAT(
            args, config, dep_tag_vocab['len'], main_tag=main_tag)  # origin network

    main_model.to(args.device)
    if sub_model is not None:
        sub_model.to(args.device)

    main_eval_results, sub_eval_results = [], []
    # Train
    if args.prune:
        if config.use_ib:
            main_eval_results, sub_eval_results = train_vib(  # VIB-SCL
                args, main_model, sub_model, train_dataset, test_dataset)
        else:
            main_eval_results, sub_eval_results = train_sdclr(  # w/o VIB
                args, main_model, sub_model, train_dataset, test_dataset)
    else:
        if config.use_ib:  # w/o CLR
            main_eval_results = train_single_vib(
                args, main_model, train_dataset, test_dataset)
        else:
            main_eval_results = train(  # origin
                args, train_dataset, main_model, test_dataset)

    if len(main_eval_results):
        best_eval_acc = max(main_eval_results, key=lambda x: x['acc'])
        best_eval_f1 = max(main_eval_results, key=lambda x: x['f1'])
        for key in sorted(best_eval_acc.keys()):
            logging.info('[Max Accuracy of Main Network]')
            logging.info(' {} = {}'.format(key, str(best_eval_acc[key])))
        for key in sorted(best_eval_f1.keys()):
            logging.info('[Max F1 of Main Network]')
            logging.info(' {} = {}'.format(key, str(best_eval_f1[key])))

    if len(sub_eval_results):
        best_eval_acc = max(sub_eval_results, key=lambda x: x['acc'])
        best_eval_f1 = max(sub_eval_results, key=lambda x: x['f1'])
        for key in sorted(best_eval_acc.keys()):
            logging.info('[Max Accuracy of Sub Network]')
            logging.info(' {} = {}'.format(key, str(best_eval_acc[key])))
        for key in sorted(best_eval_f1.keys()):
            logging.info('[Max F1 of Sub Network]')
            logging.info(' {} = {}'.format(key, str(best_eval_f1[key])))


if __name__ == "__main__":
    main()
