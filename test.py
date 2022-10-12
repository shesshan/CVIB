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
    parser.add_argument('--case_study', action='store_true',
                        help='Case study.')
    parser.add_argument('--imb_study', action='store_true',
                        help='Class Imbalance study.')
    parser.add_argument('--arts_test', action='store_true',
                        help='Robustness Test on ARTS test set.')
    parser.add_argument('--cross_domain', action='store_true',
                        help='Out-Of-Domain Generalization Test.')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to store intermedia data, such as vocab, embeddings, tags_vocab.')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of classes of ABSA.')
    parser.add_argument('--save_folder', type=str, default='./results',
                        help='Directory to store trained model.')

    parser.add_argument('--log_file', default='test.log',
                        type=str, help='location of log file')
    parser.add_argument('--config_file', default='bert_config.json',
                        type=str, help='location of BERT custom config file if specified.')

    parser.add_argument('--cuda_id', type=str, default='3',
                        help='Choose which GPUs to run')
    parser.add_argument('--seed', type=int, default=2019,
                        help='random seed for initialization')

    # Model parameters
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
    
    parser.add_argument('--add_non_connect',  type=bool, default=True,
                        help='Add a sepcial "non-connect" relation for aspect with no direct connection.')
    parser.add_argument('--multi_hop',  type=bool, default=True,
                        help='Multi hop non connection.')
    parser.add_argument('--max_hop', type=int, default=4,
                        help='max number of hops')

    parser.add_argument('--num_heads', type=int, default=6,
                        help='Number of heads for gat.')
    parser.add_argument('--final_hidden_size', type=int, default=300,
                        help='Hidden size of bilstm, in early stage.')
    parser.add_argument('--num_mlps', type=int, default=2,
                        help='Number of mlps in the last of model.')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout rate for embedding.')

    parser.add_argument('--num_gcn_layers', type=int, default=1,
                        help='Number of GCN layers.')
    parser.add_argument('--gcn_mem_dim', type=int, default=300,
                        help='Dimension of the W in GCN.')
    parser.add_argument('--gcn_dropout', type=float, default=0.2,
                        help='Dropout rate for GCN.')
    # GAT
    parser.add_argument('--gat', action='store_true',
                        help='GAT')
    parser.add_argument('--gat_our', action='store_true',
                        help='GAT_our')
    parser.add_argument('--gat_attention_type', type=str, choices=['linear', 'dotprod', 'gcn'], default='dotprod',
                        help='The attention used for gat')

    parser.add_argument('--embedding_type', type=str,
                        default='glove', choices=['glove', 'bert'])
    parser.add_argument('--glove_dir', type=str, default='/data1/mschang/glove',
                        help='Directory storing glove embeddings')
    parser.add_argument('--embedding_dim', type=int, default=300,
                        help='Dimension of glove embeddings')
    parser.add_argument('--dep_relation_embed_dim', type=int, default=300,
                        help='Dimension for dependency relation embeddings.')
    parser.add_argument('--hidden_size', type=int, default=300,
                        help='Hidden size of bilstm, in early stage.')
    parser.add_argument('--highway', action='store_true',
                        help='Use highway embed.')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers of bilstm or highway or elmo.')
    
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
                        default=False, help='self-pruning')
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

    # Testing parameters
    parser.add_argument("--per_gpu_test_batch_size", default=32, type=int,
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
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler,
                                 batch_size=args.per_gpu_test_batch_size,
                                 collate_fn=collate_fn)
    logging.info("***** Class Imbalance Testing *****")
    logging.info("  Total Num = %d", len(test_dataset))
    pos, neu, neg = 0, 0, 0
    # {'negative': 0, 'positive': 1, 'neutral': 2}
    for e in test_dataset:
        if args.gat_bert:
            label = int(e[10])
        elif args.pure_bert or args.capsnet_bert:
            label = int(e[7])
        else:
            label = int(e[6])
        #print(label)
        #assert False
        if label == 1:
            pos += 1
        elif label == 0:
            neg += 1
        elif label == 2:
            neu += 1
    logging.info(
        "Classes Num: [POS] = %d  [NEG] = %d  [NEU] = %d", pos, neg, neu)
    logging.info("  Batch Size = %d", args.per_gpu_test_batch_size)
    test_loss = 0.0
    nb_test_steps = 0

    preds = None
    out_label_ids = None
    if args.capsnet_bert:
        criterion = CapsuleLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    for batch in test_dataloader:

        model.eval()

        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs, labels = get_input_from_batch(args, batch)
            outputs = model(**inputs)
            logits = outputs[0]
        tmp_test_loss = criterion(logits, labels)
        test_loss += tmp_test_loss.mean().item()
        nb_test_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, labels.detach().cpu().numpy(), axis=0)
        #prediction = np.argmax(logits.detach().cpu().numpy(), axis=1)
        #ground_truth = labels.detach().cpu().numpy()
    
    preds = np.argmax(preds, axis=1)
    pos_pred, neg_pred, neu_pred = 0, 0, 0
    for i in range(len(out_label_ids)):
        # instances of each class that CVIB correctly classifies
        if (int(preds[i]) == int(out_label_ids[i])):
            if int(out_label_ids[i]) == 1:
                pos_pred += 1
            elif int(out_label_ids[i]) == 0:
                neg_pred += 1
            elif int(out_label_ids[i]) == 2:
                neu_pred += 1

    test_loss = test_loss / nb_test_steps
    result = compute_metrics(preds, out_label_ids)
    results.update(result)

    logging.info('***** CVIB Test Results *****')
    logging.info("  Test Loss: %s", str(test_loss))
    for key in sorted(result.keys()):
        logging.info("  %s = %s", key, str(result[key]))
    logging.info(
        "Correctly Classified Num: [POS] = %d  [NEG] = %d  [NEU] = %d", pos_pred, neg_pred, neu_pred)
    logging.info("Accuracy: [POS] = %f  [NEG] = %f  [NEU] = %f",
                 (pos_pred/pos), (neg_pred/neg), (neu_pred/neu))

    return results, test_loss


def test_case_study(args, test_dataset, model, model_2=None):
    results = {}

    test_sampler = SequentialSampler(test_dataset)
    collate_fn = get_collate_fn(args)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler,
                                 batch_size=args.per_gpu_test_batch_size,
                                 collate_fn=collate_fn)

    # Eval
    if args.arts_test:
        logging.info("***** ARTS Robustness Testing *****")
    elif args.cross_domain:
        logging.info("***** Out-of-Domain Generalization Testing *****")
    elif args.case_study:
        logging.info("***** Case Study Testing *****")

    logging.info("  Num Test Examples = %d", len(test_dataset))
    logging.info("  Batch Size = %d", args.per_gpu_test_batch_size)

    test_loss = 0.0
    nb_test_steps = 0

    preds = None
    out_label_ids = None
    preds_2 = None

    if args.capsnet_bert:
        criterion = CapsuleLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    for batch in test_dataloader:
        # for batch in tqdm(eval_dataloader, desc='Evaluating'):
        # print(batch[0])
        #assert False
        model.eval()
        if model_2 is not None:
            model_2.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs, labels = get_input_from_batch(args, batch)
            outputs = model(**inputs)
            logits = outputs[0]
            # (num_layers, num_heads, seq_len, seq_len)
            attentions = outputs[2]
            # print(len(attentions))
            # print(attentions[0].size())
            # print(inputs['input_cat_ids'].size())
            #assert False
            tmp_test_loss = criterion(logits, labels)
            test_loss += tmp_test_loss.mean().item()
            outputs_2 = model_2(**inputs)
            logits_2 = outputs_2[0]
            attentions_2 = outputs_2[2]

        nb_test_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, labels.detach().cpu().numpy(), axis=0)
        if preds_2 is None:
            preds_2 = logits_2.detach().cpu().numpy()
        else:
            preds_2 = np.append(
                preds_2, logits_2.detach().cpu().numpy(), axis=0)

        #sentences = inputs['sentence']
        prediction = np.argmax(logits.detach().cpu().numpy(), axis=1)
        prediction_2 = np.argmax(logits_2.detach().cpu().numpy(), axis=1)
        ground_truth = labels.detach().cpu().numpy()
        for i in range(len(ground_truth)):
            # instances that VIB-SCL correctly classifies while RGAT-BERT makes mistakes
            if (int(prediction[i]) == int(ground_truth[i])) and (int(prediction_2[i]) != int(ground_truth[i])):
                num_layers = len(attentions)
                batch_size, num_heads, seq_len, seq_len = attentions[0].size()
                sent_ids = inputs['input_cat_ids'][i].detach().cpu().numpy()
                aspect_ids = inputs['input_aspect_ids'][i].detach(
                ).cpu().numpy()
                sentence_text = args.tokenizer.decode(sent_ids)
                sentence_tokens = args.tokenizer.tokenize(sentence_text)
                aspect_text = args.tokenizer.decode(aspect_ids)
                logging.info('Sentence: {}, Aspect: {}, Label: {}, [VIB-SCL] Prediction: {}, [RGAT-BERT] Prediction: {} '.format(
                    sentence_text, aspect_text, int(ground_truth[i]), int(prediction[i]), int(prediction_2[i])))
                '''
            with open("case_study.txt", "w") as f:
                f.write('Sentence: {}, Aspect: {}, Label: {}, [VIB-SCL] Prediction: {}, [RGAT-BERT] Prediction: {} \n'.format(
                    sentence_text, aspect_text, int(ground_truth[i]), int(prediction[i]), int(prediction_2[i])))
                f.write(str(sentence_tokens)+'\n')
                f.write(str(sent_ids)+'\n')
                f.write('\n[VIB-SCL attention scores]\n')
                for j in range(num_heads):
                    f.write('# {}-th heads\n'.format(j))
                    for k in range(seq_len):
                        f.write('{}-th token\n'.format(k))
                        f.write(str(attentions[i][0][j][k])+'\n')
                f.write('[RGAT-BERT attention scores]\n')
                for j in range(num_heads):
                    f.write('# {}-th heads\n'.format(j))
                    for k in range(seq_len):
                        f.write('{}-th token\n'.format(k))
                        f.write(str(attentions_2[i][0][j][k])+'\n')
            f.close()
                '''
                with open("case_study_vibscl_5.csv", "w") as f:
                    for token in sentence_tokens:
                        f.write(str(token)+',')
                    f.write('\n')
                    for m in range(num_layers):
                        sum_all_heads = attentions[m][i].sum(
                            dim=0)  # (seq_len, seq_len)
                        for j in range(seq_len):
                            for k in range(seq_len):
                                f.write(str(float(sum_all_heads[j][k]))+',')
                            f.write('\n')
                        f.write('\n')
                f.close()
                with open("case_study_rgat_5.csv", "w") as f:
                    for token in sentence_tokens:
                        f.write(str(token)+',')
                    f.write('\n')
                    for m in range(num_layers):
                        sum_all_heads = attentions_2[m][i].sum(
                            dim=0)  # (seq_len, seq_len)
                        for j in range(seq_len):
                            for k in range(seq_len):
                                f.write(str(float(sum_all_heads[j][k]))+',')
                            f.write('\n')
                        f.write('\n')
                f.close()

            '''
            if (int(prediction[i]) == int(ground_truth[i])) and (int(prediction_2[i]) != int(ground_truth[i])):
                # instances that VIB-SCL correctly classifies while RGAT-BERT makes mistakes
                sent_ids = inputs['input_ids'][i].detach().cpu().numpy()
                aspect_ids = inputs['input_aspect_ids'][i].detach(
                ).cpu().numpy()
                sentence_text = args.tokenizer.decode(sent_ids)
                aspect_text = args.tokenizer.decode(aspect_ids)
                logging.info('Sentence: {}, Aspect: {}, Label: {}, [VIB-SCL] Prediction: {}, [RGAT-BERT] Prediction: {} '.format(
                    sentence_text, aspect_text, int(ground_truth[i]), int(prediction[i]), int(prediction_2[i])))
            '''

    test_loss = test_loss / nb_test_steps
    preds = np.argmax(preds, axis=1)
    result = compute_metrics(preds, out_label_ids)
    results.update(result)

    logging.info('***** CVIB Test Results *****')
    logging.info("  Test Loss: %s", str(test_loss))
    for key in sorted(result.keys()):
        logging.info("  %s = %s", key, str(result[key]))

    preds_2 = np.argmax(preds_2, axis=1)
    result = compute_metrics(preds_2, out_label_ids)
    logging.info('***** RGAT-BERT Test Results *****')
    for key in sorted(result.keys()):
        logging.info("  %s = %s", key, str(result[key]))

    return results, test_loss


def load_checkpoint(args, model, main_tag=False, vib_tag=False):
    """load pre-train model for representation learning.
    """
    if not os.path.exists(str(args.save_folder)):
        os.mkdir(str(args.save_folder))
    save_path = str(args.save_folder)+'/checkpoints'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if args.prune:
        if vib_tag:
            save_path = save_path + '/CLR-VIB'
            #masks, prune_stat = model.get_masks()
        else:
            save_path = save_path + '/only-CLR'
    else:
        if vib_tag:
            save_path = save_path+'/only-VIB'
        else:
            save_path = save_path + '/pure'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if main_tag:
        model_pt = 'origin_model.pt'
    else:
        model_pt = 'pruned_model.pt'
    file_path = os.path.join(save_path, model_pt)
    checkpoint = torch.load(file_path, map_location="cpu")
    model_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(model_state_dict)
    logging.info('Load model from [{}] at global training step [{}]'.format(
        file_path, checkpoint['global_step']))
    return model


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
    # adding more baseline models here to verify the efficiency of our framework...

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    save_folder += '/' + \
        str(args.source_domain)+'-'+str(args.target_domain)

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
    if args.arts_test:
        args.arts_test = False
        _, _, _, dep_tag_vocab, _ = load_datasets_and_vocabs(
            args)
        args.arts_test = True
        _, test_dataset, _, _, _ = load_datasets_and_vocabs(
            args)
    elif args.cross_domain:
        target_domain = args.target_domain
        args.target_domain = args.source_domain
        _, _, _, dep_tag_vocab, _ = load_datasets_and_vocabs(
            args)
        args.target_domain = target_domain
        _, test_dataset, _, _, _ = load_datasets_and_vocabs(
            args)
    else:
        _, test_dataset, _, dep_tag_vocab, _ = load_datasets_and_vocabs(
            args)

    # config_file can be customized json file or pretrain-model load path
    if args.config_file:
        config = BertConfig.from_pretrained(args.config_file)
    else:
        config = BertConfig.from_pretrained(args.bert_model_dir)

    if config.use_ib:
        main_tag = False  # VIB
    else:
        main_tag = True  # no-VIB

    # Build Model
    # model = Aspect_Text_Multi_Syntax_Encoding(args, dep_tag_vocab['len'], pos_tag_vocab['len'])
    if args.gat_our:
        model = Aspect_Relational_GAT(
            args=args, dep_tag_num=dep_tag_vocab['len'])
    elif args.pure_bert:
        model = Pure_Bert(args, config, main_tag=main_tag)
    elif args.gat_bert:
        model = Aspect_Bert_GAT(
            args, config, dep_tag_vocab['len'], main_tag=main_tag)
    elif args.capsnet_bert:
        model = CapsNet_Bert(args=args, config=config, main_tag=main_tag)
    model.to(args.device)
    model_2 = None
    if args.case_study:
        model_2 = Aspect_Bert_GAT(
            args, config, dep_tag_vocab['len'], main_tag=True)
        model_2.to(args.device)

    if args.prune:
        if config.use_ib:
            model = load_checkpoint(  # CLR-VIB
                args, model, main_tag=False, vib_tag=True)
        else:
            model = load_checkpoint(  # only-CLR
                args, model, main_tag=False, vib_tag=False)
    else:
        if config.use_ib:
            model = load_checkpoint(  # only-VIB
                args, model, main_tag=True, vib_tag=True)
        else:
            model = load_checkpoint(  # pure
                args, model, main_tag=True, vib_tag=False)

    if args.case_study:
        args.prune = False
        model_2 = load_checkpoint(
            args, model_2, main_tag=True, vib_tag=False)
        args.prune = True

    if args.case_study:
        results, _ = test_case_study(args, test_dataset, model, model_2)
    elif args.imb_study:
        results, _ = test_imb_study(args, test_dataset, model)

    logging.info('[Test Accuracy]')
    logging.info(results['acc'])
    logging.info('[Test F1]')
    logging.info(results['f1'])


if __name__ == "__main__":
    main()
