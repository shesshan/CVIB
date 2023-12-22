import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from trainer import *
import argparse
from model import *
from custom_datasets import load_datasets_and_vocabs
from transformers import BertTokenizer, BertConfig

MODELS = {'rgat_bert': Aspect_Bert_GAT}

ARTS_SETS = ['ALL', 'REVTGT', 'REVNON', 'ADDDIFF']

sentiment_lookup = {'negative': 0, 'positive': 1, 'neutral': 2}
reversed_sentiment_lookup = {v: k for k, v in sentiment_lookup.items()}


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
    parser.add_argument('--arts_set',
                        type=str,
                        default='ALL',
                        choices=ARTS_SETS,
                        help='ARTS datasets for testing.')
    parser.add_argument('--cross_domain',
                        action='store_true',
                        help='Out-Of-Domain Generalization Test.')
    parser.add_argument('--case_study',
                        action='store_true',
                        help='Case study.')
    parser.add_argument('--imb_study',
                        action='store_true',
                        help='Long-tail Test.')
    parser.add_argument('--bad_case_analysis',
                        action='store_true',
                        help='Bad Cases Analysis.')
    parser.add_argument(
        '--data_root_dir',
        type=str,
        default='./ABSA_RGAT',
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
                        default='0',
                        help='Choose which GPUs to run')
    parser.add_argument('--seed',
                        type=int,
                        default=2023,
                        help='random seed for initialization')

    # Model parameters
    parser.add_argument('--embedding_type',
                        type=str,
                        default='bert',
                        choices=['glove', 'bert'])
    parser.add_argument('--glove_dir',
                        type=str,
                        default='./glove',
                        help='Directory of GloVe embeddings.')
    parser.add_argument('--embedding_dim',
                        type=int,
                        default=300,
                        help='GloVe embeddings dimension.')
    parser.add_argument('--bert_model_dir',
                        type=str,
                        default='bert-base-uncased',
                        help='Path to pre-trained Bert model.')
    parser.add_argument('--model_name',
                        type=str,
                        default='rgat_bert',
                        choices=['asgcn_bert', 'rgat_bert'],
                        help='backbone model.')
    parser.add_argument('--spc',
                        action='store_true',
                        default=False,
                        help='use sentence-aspect pair as input.')

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
                        help='Number of heads for gat.')
    parser.add_argument('--final_hidden_size',
                        type=int,
                        default=300,
                        help='Hidden size of bilstm, in early stage.')
    parser.add_argument('--num_mlps',
                        type=int,
                        default=2,
                        help='Number of mlps in the last of model.')
    parser.add_argument('--dropout',
                        type=float,
                        default=0,
                        help='Dropout rate for embedding.')
    # GAT
    parser.add_argument('--gat', action='store_true', help='GAT')
    parser.add_argument('--gat_our', action='store_true', help='GAT_our')
    parser.add_argument('--gat_attention_type',
                        type=str,
                        choices=['linear', 'dotprod', 'gcn'],
                        default='dotprod',
                        help='The attention used for gat')

    parser.add_argument('--hidden_size',
                        type=int,
                        default=300,
                        help='Hidden size of bilstm, in early stage.')
    parser.add_argument('--highway',
                        action='store_true',
                        help='Use highway embed.')
    parser.add_argument('--num_layers',
                        type=int,
                        default=2,
                        help='Number of layers of bilstm or highway or elmo.')
    parser.add_argument('--gat_dropout',
                        type=float,
                        default=0,
                        help='Dropout rate for GAT.')
    # ASGCN
    parser.add_argument('--num_gcn_layers',
                        type=int,
                        default=1,
                        help='Number of GCN layers.')
    parser.add_argument('--gcn_dropout',
                        type=float,
                        default=0.3,
                        help='Dropout rate for GCN.')
    parser.add_argument('--max_seq_len',
                        type=int,
                        default=128,
                        help='max length of input sequences.')

    # SCL
    parser.add_argument('--prune',
                        action='store_true',
                        default=False,
                        help='self-pruning')
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

    # Testing parameters
    parser.add_argument("--per_gpu_test_batch_size",
                        default=32,
                        type=int,
                        help="Batch size per GPU/CPU for test.")

    return parser.parse_args()


def check_args(args):
    '''
    eliminate confilct situations
    '''
    logging.info(vars(args))


def test_imb_study(args, test_dataset, model):
    results = {}
    test_sampler = SequentialSampler(test_dataset)
    collate_fn = get_collate_fn(args)
    test_dataloader = DataLoader(test_dataset,
                                 sampler=test_sampler,
                                 batch_size=args.per_gpu_test_batch_size,
                                 collate_fn=collate_fn)
    logging.info("***** Class Imbalance Testing *****")
    logging.info("  Total Num = %d", len(test_dataset))
    pos, neu, neg = 0, 0, 0
    for e in test_dataset:
        if 'gcn' in args.model_name:
            label = int(e[7])
        elif 'gat' in args.model_name:
            label = int(e[14])

        if label == 1:
            pos += 1
        elif label == 0:
            neg += 1
        elif label == 2:
            neu += 1

    logging.info("Classes Num: [POS] = %d  [NEG] = %d  [NEU] = %d", pos, neg,
                 neu)
    logging.info("  Batch Size = %d", args.per_gpu_test_batch_size)

    preds = None
    out_label_ids = None
    criterion = nn.CrossEntropyLoss()

    for batch in test_dataloader:
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs, labels = get_input_from_batch(args, batch)
            outputs = model(**inputs)
            logits = outputs[0]

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids,
                                      labels.detach().cpu().numpy(),
                                      axis=0)

    preds = np.argmax(preds, axis=1)
    pos_pred, neg_pred, neu_pred = 0, 0, 0
    for i in range(len(out_label_ids)):
        if (int(preds[i]) == int(out_label_ids[i])):
            if int(out_label_ids[i]) == 1:
                pos_pred += 1
            elif int(out_label_ids[i]) == 0:
                neg_pred += 1
            elif int(out_label_ids[i]) == 2:
                neu_pred += 1

    result = compute_metrics(preds, out_label_ids)
    results.update(result)

    logging.info('***** CVIB Long-tail Results *****')
    for key in sorted(result.keys()):
        logging.info("  %s = %s", key, str(result[key]))
    logging.info(
        "Correctly Classified Num: [POS] = %d  [NEG] = %d  [NEU] = %d",
        pos_pred, neg_pred, neu_pred)
    logging.info("Accuracy: [POS] = %f  [NEG] = %f  [NEU] = %f",
                 (pos_pred / pos), (neg_pred / neg), (neu_pred / neu))

    return results


def bad_case_analysis(args, test_dataset, model):

    test_sampler = SequentialSampler(test_dataset)
    if 'gcn' in args.model_name:
        test_dataloader = DataLoader(test_dataset,
                                     sampler=test_sampler,
                                     batch_size=args.per_gpu_test_batch_size)
    else:
        collate_fn = get_collate_fn(args)
        test_dataloader = DataLoader(test_dataset,
                                     sampler=test_sampler,
                                     batch_size=args.per_gpu_test_batch_size,
                                     collate_fn=collate_fn)
    # Eval
    print("***** Bad Case Testing *****")
    print(" Testing Examples = {}".format(len(test_dataset)))
    print(" Testing Batch Size = {}".format(args.per_gpu_test_batch_size))

    preds = None
    out_label_ids = None

    for batch in test_dataloader:
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs, labels = get_input_from_batch(args, batch)
            outputs = model(**inputs)
            logits = outputs[0]
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids,
                                      labels.detach().cpu().numpy(),
                                      axis=0)
        prediction = np.argmax(logits.detach().cpu().numpy(), axis=1)
        ground_truth = labels.detach().cpu().numpy()

        bad_case_list = []
        for i in range(len(ground_truth)):
            if int(prediction[i]) != int(ground_truth[i]):
                sent_ids = inputs['input_cat_ids'][i].detach().cpu().numpy()
                aspect_ids = inputs['input_aspect_ids'][i].detach().cpu(
                ).numpy()
                sentence_text = args.tokenizer.decode(sent_ids)
                aspect_text = args.tokenizer.decode(aspect_ids)
                records = 'Sentence: {}, Aspect: {}, Label: {}, [RGAT-BERT] Prediction: {} '.format(
                    sentence_text, aspect_text,
                    reversed_sentiment_lookup[int(ground_truth[i])],
                    reversed_sentiment_lookup[int(prediction[i])])
                bad_case_list.append(records)

        with open("bad_cases.txt", "a") as f:
            for i in bad_case_list:
                f.write(i + '\n')
        f.close()
    return


def test_case_study(args,
                    test_dataset,
                    model,
                    model_2=None:
    results = {}

    test_sampler = SequentialSampler(test_dataset)
    collate_fn = get_collate_fn(args)
    test_dataloader = DataLoader(test_dataset,
                                 sampler=test_sampler,
                                 batch_size=args.per_gpu_test_batch_size,
                                 collate_fn=collate_fn)

    print("***** Case Study Testing *****")
    print(" Testing Examples = {}".format(len(test_dataset)))
    print(" Testing Batch Size = {}".format(args.per_gpu_test_batch_size))

    preds = None
    out_label_ids = None
    preds_2 = None

    for batch in test_dataloader:
        model.eval()
        model_2.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs, labels = get_input_from_batch(args, batch)
            outputs = model(**inputs)
            logits = outputs[0]
            # (num_layers, num_heads, seq_len, seq_len)
            
            attentions = outputs[2]
            outputs_2 = model_2(**inputs)
            logits_2 = outputs_2[0]
            attentions_2 = outputs_2[2]

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids,
                                      labels.detach().cpu().numpy(),
                                      axis=0)

        if preds_2 is None:
            preds_2 = logits_2.detach().cpu().numpy()
        else:
            preds_2 = np.append(preds_2,
                                logits_2.detach().cpu().numpy(),
                                axis=0)

    prediction = np.argmax(logits.detach().cpu().numpy(), axis=1)
    prediction_2 = np.argmax(logits_2.detach().cpu().numpy(), axis=1)
    ground_truth = labels.detach().cpu().numpy()

    bad_case_list = []
    for i in range(len(ground_truth)):
        # instances that CVIB correctly classifies while RGAT-BERT fails
        if (int(prediction[i]) == int(ground_truth[i])) and (int(
                prediction_2[i]) != int(ground_truth[i])):
            num_layers = len(attentions)
            batch_size, num_heads, seq_len, seq_len = attentions[0].size()
            sent_ids = inputs['input_cat_ids'][i].detach().cpu().numpy()
            aspect_ids = inputs['input_aspect_ids'][i].detach().cpu().numpy()
            sentence_text = args.tokenizer.decode(sent_ids)
            sentence_tokens = args.tokenizer.tokenize(sentence_text)
            aspect_text = args.tokenizer.decode(aspect_ids)
            records = 'Sentence: {}, Aspect: {}, Label: {}, [CVIB] Prediction: {}, [RGAT-BERT] Prediction: {} '.format(
                sentence_text, aspect_text, int(ground_truth[i]),
                int(prediction[i]), int(prediction_2[i]))
            bad_case_list.append(records)
            logging.info(records)

        with open("case_study.txt", "w") as f:
            for i in bad_case_list:
                f.write(i + '\n')
        f.close()

    result = compute_metrics(prediction, out_label_ids)
    results.update(result)

    logging.info('***** CVIB Testing Results *****')
    for key in sorted(result.keys()):
        logging.info("  %s = %s", key, str(result[key]))

    result = compute_metrics(prediction_2, out_label_ids)
    logging.info('***** RGAT-BERT Testing Results *****')
    for key in sorted(result.keys()):
        logging.info("  %s = %s", key, str(result[key]))

    return results


def test_arts(args, test_datasets, model):
    '''
    ARTS Robustness Testing
    '''
    results = []
    sets = ARTS_SETS
    print("***** ARTS Robustness Testing *****")
    for i in range(len(test_datasets)):
        test_dataset = test_datasets[i]
        test_sampler = SequentialSampler(test_dataset)
        if 'gcn' in args.model_name:
            test_dataloader = DataLoader(
                test_dataset,
                sampler=test_sampler,
                batch_size=args.per_gpu_test_batch_size)
        else:
            collate_fn = get_collate_fn(args)
            test_dataloader = DataLoader(
                test_dataset,
                sampler=test_sampler,
                batch_size=args.per_gpu_test_batch_size,
                collate_fn=collate_fn)

        print("[{}] Testing Samples = {}".format(sets[i], len(test_dataset)))

        preds = None
        out_label_ids = None
        for batch in test_dataloader:
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs, labels = get_input_from_batch(args, batch)
                outputs = model(**inputs)
                logits = outputs[0]
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids,
                                          labels.detach().cpu().numpy(),
                                          axis=0)

        preds = np.argmax(preds, axis=1)
        result = compute_metrics(preds, out_label_ids)
        print('[{}] Testing Results:'.format(sets[i]))
        for k, v in result.items():
            print('{} = {}'.format(k, v))
        results.append(result)


    return results


def load_checkpoint(args, model, main_tag=False):
    """load pre-train model for representation learning.
    """
    save_path = str(args.save_folder) + '/checkpoints'

    if args.prune:
        if args.use_ib:
            save_path = save_path + '/CVIB'
        else:
            save_path = save_path + '/only-SCL'
    else:
        if args.use_ib:
            save_path = save_path + '/only-VIB'
        else:
            save_path = save_path + '/pure'

    if main_tag:
        model_pt = 'origin_model.pt'
    else:
        model_pt = 'pruned_model.pt'

    file_path = os.path.join(save_path, model_pt)
    print(file_path)
    if not os.path.exists(save_path):
        raise ValueError('cannot load model checkpoint!')
    checkpoint = torch.load(file_path, map_location="cpu")
    model_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(model_state_dict)
    return model


def main():

    # Parse args
    args = parse_args()
    check_args(args)

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    save_folder = '{}/{}'.format(args.save_folder, args.model_name)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    save_folder = '{}/{}-{}'.format(
        save_folder, args.source_domain,
        args.target_domain + '-ARTS' if args.arts_test else args.target_domain)

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
    logging.info('Evaluating Device: %s', str(args.device).upper())
    # Set seed
    set_seed(args)

    # Bert, load pretrained model and tokenizer, check if neccesary to put bert here
    if args.embedding_type == 'bert':  # embedding_type: glove OR bert
        tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir)
        
    # Load datasets and vocabs
    if args.arts_test:
        test_datasets = []
        args.arts_set = "ALL"
        _, test_dataset, _, dep_tag_vocab, _ = load_datasets_and_vocabs(args)
        test_datasets.append(test_dataset)
        for i in ARTS_SETS[1:]:
            args.arts_set = i
            _, test_dataset, _, _, _ = load_datasets_and_vocabs(args)
            test_datasets.append(test_dataset)
    elif args.cross_domain:
        target_domain = args.target_domain
        args.target_domain = args.source_domain
        _, _, _, dep_tag_vocab, _ = load_datasets_and_vocabs(args)
        args.target_domain = target_domain
        _, test_dataset, _, _, _ = load_datasets_and_vocabs(args)
    else:
        _, test_dataset, _, dep_tag_vocab, _ = load_datasets_and_vocabs(args)

    # config_file can be customized json file or pretrain-model load path
    if args.config_file:
        config = BertConfig.from_pretrained(args.config_file)
        print('Load customized config file.')
    else:
        config = BertConfig.from_pretrained(args.bert_model_dir)

    # Build Model
    model = MODELS[args.model_name]
    if 'rgat' in args.model_name:
        model = model(args, config, dep_tag_vocab['len'], main_tag=False)
    else:
        model = model(args, config, main_tag=False)
    model.to(device)
    model = load_checkpoint( 
                args, model, main_tag=False)

    model_2 = None
    if args.case_study:
        # set model_2 (RGAT-BERT)
        model_2 = Aspect_Bert_GAT(args,
                                  config,
                                  dep_tag_vocab['len'],
                                  main_tag=True)
        model_2.to(args.device)
        args.prune = False
        args.use_ib = False
        model_2 = load_checkpoint(args, model_2, main_tag=True)

    args.prune = True
    args.use_ib = True


    if args.case_study:
        results, _ = test_case_study(args, test_dataset, model, model_2)
    elif args.imb_study:
        results, _ = test_imb_study(args, test_dataset, model)
    elif args.arts_test:
        results = test_arts(args, test_datasets, model)
    elif args.bad_case_analysis:
        bad_case_analysis(args, test_dataset, model)


if __name__ == "__main__":
    main()
