from ast import arg
import logging
import os
import random
from utils.losses import SdConLoss, SupConLoss, Similarity, CapsuleLoss
from model_utils import SparsePruner
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, matthews_corrcoef
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.distributed as dist
from tqdm import tqdm, trange

from custom_datasets import my_collate, my_collate_elmo, my_collate_bert
from transformers import AdamW

logger = logging.getLogger(__name__)



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def get_input_from_batch(args, batch):
    embedding_type = args.embedding_type
    if embedding_type == 'glove' or embedding_type == 'elmo':
        inputs = {
            'sentence': batch[0],  # sentence_ids
            'aspect': batch[1],  # aspect token
            'dep_tags': batch[2],  # reshaped
            'pos_class': batch[3],
            'text_len': batch[4],
            'aspect_len': batch[5],
            'dep_rels': batch[7],  # adj no-reshape
            'dep_heads': batch[8],
            'aspect_position': batch[9],
            'dep_dirs': batch[10]
        }
        labels = batch[6]
    else:  # bert
        if 'gcn' in args.model_name:
            inputs = {
                'input_cat_ids': batch[0], 
                'segment_ids': batch[1], 
                'text_len': batch[2], 
                'aspect_len': batch[3],
                'left_len':batch[4], 
                'adj':batch[5], 
                'word_indexer':batch[6]
            }
            labels = batch[7]
        elif 't5' in args.model_name:
            inputs = {
                'input_ids': batch[0]
            }
            labels = batch[14]
        else:
            inputs = {
                'input_ids': batch[0],
                'input_cat_ids': batch[1],
                'segment_ids': batch[2],
                'word_indexer': batch[3],
                'input_aspect_ids': batch[4],
                'aspect_indexer': batch[5],
                'dep_tags': batch[6],  # i.e. dep_tag_ids
                'pos_class': batch[7],
                'text_len': batch[8],
                'aspect_len': batch[9],
                'dep_rels': batch[10],
                'dep_heads': batch[11],
                'aspect_position': batch[12],
                'dep_dirs': batch[13]
            }
            labels = batch[14]  # i.e. sentiment
        
    return inputs, labels


def get_collate_fn(args):
    embedding_type = args.embedding_type
    if embedding_type == 'glove':
        return my_collate
    elif embedding_type == 'elmo':
        return my_collate_elmo
    else:
        return my_collate_bert


def get_bert_optimizer(args, model, main_tag=True):

    no_decay = ['bias', 'LayerNorm.weight']

    if args.ib_lr == -1:
        args.ib_lr = args.lr # if not specified, keep it the same as args.lr
    if args.ib_wd == -1:
        args.ib_wd = args.weight_decay
    
    ib_param_list, ib_name_list, bert_param_list, bert_name_list, nd_param_list, nd_name_list = [], [], [], [], [], []
    for name, param in model.named_parameters():
        if 'z_mu' in name or 'z_logD' in name:
            ib_param_list.append(param)
            ib_name_list.append(name)
        elif any(nd in name for nd in no_decay):
            nd_param_list.append(param)
            nd_name_list.append(name)
        else:
            bert_param_list.append(param)
            bert_name_list.append(name)

    logging.info(
            '[BERT Layers] learning rate: {}, weight decay: {}'.format(
                args.lr, args.weight_decay))
    logging.info('[No-Decay Params] learning rate: {}'.format(
            args.nd_lr))
    if not main_tag:
        logging.info(
            '[VIB-based Masking Layers] learning rate: {}, weight decay: {}'
            .format(args.ib_lr, args.ib_wd))
        param_groups = [{
            'params': ib_param_list,
            'lr': args.ib_lr,
            'weight_decay': args.ib_wd
        }, {
            'params': bert_param_list,
            'lr': args.lr,
            'weight_decay': args.weight_decay
        }, {
            'params': nd_param_list,
            'lr': args.nd_lr,
            'weight_decay': 0.0
        }]
    else:
        param_groups = [{
            'params': bert_param_list,
            'lr': args.lr,
            'weight_decay': args.weight_decay
        }, {
            'params': nd_param_list,
            'lr': args.nd_lr,
            'weight_decay': 0.0
        }]

    return args.optimizer(param_groups, eps=args.adam_epsilon)


def gatherFeatures(features):
    # Dummy vectors for allgather
    features_list = [
        torch.zeros_like(features) for _ in range(dist.get_world_size())
    ]
    # Allgather
    dist.all_gather(tensor_list=features_list, tensor=features.contiguous())

    # Since allgather results do not have gradients, we replace the
    # current process's corresponding embeddings with original tensors
    features_list[dist.get_rank()] = features
    features = torch.cat(features_list, 0)
    return features


def train_vib(args, main_model, sub_model, train_dataset, test_dataset):
    '''
    Training CVIB Framework
    '''
    tb_writer = SummaryWriter(comment='_cvib_' +
                              str(args.source_domain))
    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    
    if 'gcn' in args.model_name:
        train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size)
    else:
        collate_fn = get_collate_fn(args)
        train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader
        ) // args.gradient_accumulation_steps * args.num_train_epochs

    main_optimizer = get_bert_optimizer(args, main_model)
    sub_optimizer = get_bert_optimizer(args, sub_model, main_tag=False)

    logging.info(
        " [SCL Loss] alpha factor: {}, temperature: {}".
        format(args.sdcl_fac, args.sd_temp))
    logging.info(" [VIB loss] alpha factor: {}".format(args.kl_fac))
    # Train
    logging.info(
        "***** Contrastive Variational Information Bottleneck (CVIB) Training *****"
    )
    logging.info("  Training samples = %d", len(train_dataset))
    logging.info("  Epochs = %d", args.num_train_epochs)
    logging.info("  Training batch size per GPU = %d",
                 args.per_gpu_train_batch_size)
    logging.info("  Gradient accumulation steps = %d",
                 args.gradient_accumulation_steps)
    logging.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_main_loss, main_logging_loss, tr_main_ce_loss, main_ce_logging_loss, tr_main_sdcl_loss, main_sdcl_logging_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    tr_sub_loss, sub_logging_loss, tr_sub_ce_loss, sub_ce_logging_loss, tr_sub_kl_loss, sub_kl_logging_loss, tr_sub_sdcl_loss, sub_sdcl_logging_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    main_eval_results, sub_eval_results = [], []

    main_model.zero_grad()
    sub_model.zero_grad()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)

    # [6.7 add]
    CELoss = nn.CrossEntropyLoss()
    SDLoss = SdConLoss(args)
    main_max_acc, main_max_f1, sub_max_acc, sub_max_f1 = 0.0, 0.0, 0.0, 0.0
    for epoch in train_iterator:
    
        sub_optimizer.param_groups[0]['lr'] = args.ib_lr * \
            (args.lr_fac ** (epoch//args.lr_epoch))  # decreasing LR of VIB layers

        epoch_iterator = tqdm(train_dataloader,
                              desc='Train Ep. #{}: '.format(epoch),
                              total=len(train_dataloader),
                              disable=False,
                              ascii=True)

        for step, batch in enumerate(train_dataloader):
            main_model.train()
            sub_model.train()

            main_optimizer.zero_grad()
            sub_optimizer.zero_grad()

            main_ce_loss, main_sdcl_loss, sub_ce_loss, kl_loss, sub_sdcl_loss = 0.0, 0.0, 0.0, 0.0, 0.0

            # 1. calculate the grad for original network
            with torch.no_grad():
                # pruned network
                batch = tuple(t.to(args.device) for t in batch)
                inputs, labels = get_input_from_batch(args, batch)
                outputs = sub_model(**inputs)
                logits = outputs[0]
                features = outputs[1]
                features_sub_no_grad = [
                    gatherFeatures(features).detach()
                    if dist.is_initialized() else features.detach()
                ][0]

            batch = tuple(t.to(args.device) for t in batch)
            inputs, labels = get_input_from_batch(args, batch)
            outputs = main_model(**inputs)
            logits = outputs[0]
            features = outputs[1]

            main_ce_loss = CELoss(logits, labels)
            #cl_loss = CLLoss(features_1, labels)

            features_main = [
                gatherFeatures(features) if dist.is_initialized() else features
            ][0]
            main_sdcl_loss = SDLoss(
                features_sub_no_grad.unsqueeze(1),  # (bs,1,hidden_size)
                features_main.unsqueeze(0))  # (1,bs,hidden_size)

            main_loss = main_ce_loss + args.sdcl_fac * main_sdcl_loss

            if args.gradient_accumulation_steps > 1:
                main_loss = main_loss / args.gradient_accumulation_steps

            main_loss.backward()
            torch.nn.utils.clip_grad_norm_(main_model.parameters(),
                                           args.max_grad_norm)

            main_loss_cpu = main_loss.detach().cpu()
            main_ce_loss_cpu = main_ce_loss.detach().cpu()
            main_sdcl_loss_cpu = (args.sdcl_fac *
                                  main_sdcl_loss).detach().cpu()

            tr_main_loss += main_loss_cpu.item()
            tr_main_ce_loss += main_ce_loss_cpu.item()
            tr_main_sdcl_loss += main_sdcl_loss_cpu.item()

            # 2. calculate the grad for self-pruned network
            features_main_no_grad = features_main.detach()
            batch = tuple(t.to(args.device) for t in batch)
            inputs, labels = get_input_from_batch(args, batch)
            outputs = sub_model(**inputs)
            logits = outputs[0]
            features = outputs[1]
            kl_loss = outputs[2]

            sub_ce_loss = CELoss(logits, labels)
            features_sub = [
                gatherFeatures(features) if dist.is_initialized() else features
            ][0]
            sub_sdcl_loss = SDLoss(features_sub.unsqueeze(1),
                                   features_main_no_grad.unsqueeze(0))

            sub_loss = sub_ce_loss + args.kl_fac * kl_loss + args.sdcl_fac * sub_sdcl_loss

            if args.gradient_accumulation_steps > 1:
                sub_loss = sub_loss / args.gradient_accumulation_steps

            sub_loss.backward()
            torch.nn.utils.clip_grad_norm_(sub_model.parameters(),
                                           args.max_grad_norm)

            sub_loss_cpu = sub_loss.detach().cpu()
            sub_ce_loss_cpu = sub_ce_loss.detach().cpu()
            kl_loss_cpu = (args.kl_fac * kl_loss).detach().cpu()
            sub_sdcl_loss_cpu = (args.sdcl_fac * sub_sdcl_loss).detach().cpu()

            tr_sub_loss += sub_loss_cpu.item()
            tr_sub_ce_loss += sub_ce_loss_cpu.item()
            tr_sub_kl_loss += kl_loss_cpu.item()
            tr_sub_sdcl_loss += sub_sdcl_loss_cpu.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # scheduler.step()  # Update learning rate schedule
                main_optimizer.step()
                main_model.zero_grad()

                sub_optimizer.step()
                sub_model.zero_grad()

                global_step += 1

                # Log metrics
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Eval main network
                    results, eval_loss = evaluate(args, test_dataset,
                                                  main_model)
                    # Save main network
                    if results['acc'] > main_max_acc:
                        main_max_acc = results['acc']
                        logging.info(
                            'max accuracy of origin network: {}'.format(
                                main_max_acc))
                        save_checkpoint(args,
                                        main_model,
                                        global_step,
                                        main_optimizer,
                                        main_tag=True)
                    main_eval_results.append(results)
                    for key, value in results.items():
                        tb_writer.add_scalar('eval_{}'.format(key), value,
                                             global_step)
                    tb_writer.add_scalar('eval_loss', eval_loss, global_step)
                    # Eval sub network
                    results, eval_loss = evaluate(args, test_dataset,
                                                  sub_model)
                    # Save sub network
                    if results['acc'] > sub_max_acc:
                        sub_max_acc = results['acc']
                        logging.info(
                            'max accuracy of pruned network: {}'.format(
                                sub_max_acc))
                        save_checkpoint(args,
                                        sub_model,
                                        global_step,
                                        sub_optimizer,
                                        main_tag=False)
                    sub_eval_results.append(results)
                    for key, value in results.items():
                        tb_writer.add_scalar('sub_eval_{}'.format(key), value,
                                             global_step)
                    tb_writer.add_scalar('sub_eval_loss', eval_loss,
                                         global_step)

                    tb_writer.add_scalar('main_train_loss',
                                         (tr_main_loss - main_logging_loss) /
                                         args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'main_ce_loss',
                        (tr_main_ce_loss - main_ce_logging_loss) /
                        args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'main_sdcl_loss',
                        (tr_main_sdcl_loss - main_sdcl_logging_loss) /
                        args.logging_steps, global_step)
                    tb_writer.add_scalar('sub_train_loss',
                                         (tr_sub_loss - sub_logging_loss) /
                                         args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'sub_ce_loss', (tr_sub_ce_loss - sub_ce_logging_loss) /
                        args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'sub_kl_loss', (tr_sub_kl_loss - sub_kl_logging_loss) /
                        args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'sub_sdcl_loss',
                        (tr_sub_sdcl_loss - sub_sdcl_logging_loss) /
                        args.logging_steps, global_step)

                    main_logging_loss = tr_main_loss
                    main_ce_logging_loss = tr_main_ce_loss
                    main_sdcl_logging_loss = tr_main_sdcl_loss
                    sub_logging_loss = tr_sub_loss
                    sub_ce_logging_loss = tr_sub_ce_loss
                    sub_kl_logging_loss = tr_sub_kl_loss
                    sub_sdcl_logging_loss = tr_sub_sdcl_loss

                    masks, prune_stat = sub_model.get_masks()
                    logging.info(prune_stat)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            epoch_iterator.close()
            break

    tb_writer.close()
    return main_eval_results, sub_eval_results


def evaluate(args, eval_dataset, model):
    results = {}

    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    if 'gcn' in args.model_name:
        eval_dataloader = DataLoader(eval_dataset,
                                 sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,)
    else:
        collate_fn = get_collate_fn(args)
        eval_dataloader = DataLoader(eval_dataset,
                                 sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)
    

    # Eval
    logging.info("***** Running Evaluation *****")
    logging.info("  Num examples = %d", len(eval_dataset))
    logging.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    CELoss = nn.CrossEntropyLoss()
    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs, labels = get_input_from_batch(args, batch)
            outputs = model(**inputs)
            logits = outputs[0]
            tmp_eval_loss = CELoss(logits, labels)
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids,
                                      labels.detach().cpu().numpy(),
                                      axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    # print(preds)
    result = compute_metrics(preds, out_label_ids)
    results.update(result)

    logging.info('***** Evaluation Results *****')
    logging.info("  eval loss: %s", str(eval_loss))
    for key in sorted(result.keys()):
        logging.info("  %s = %s", key, str(result[key]))

    return results, eval_loss


def save_checkpoint(args,
                    model,
                    global_step,
                    optimizer=None,
                    main_tag=False):
    """Saves model to checkpoints."""
    save_path = str(args.save_folder) + '/checkpoints'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if main_tag:
        file_path = os.path.join(save_path, 'origin_model.pt')
    else:
        file_path = os.path.join(save_path, 'pruned_model.pt')
        _, prune_stat = model.get_masks()

    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    torch.save(checkpoint, file_path)
    logging.info('Save model to [{}].'.format(file_path))


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {"acc": acc, "f1": f1}


def compute_metrics(preds, labels):
    return acc_and_f1(preds, labels)
