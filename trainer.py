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

from custom_datasets import my_collate, my_collate_elmo, my_collate_pure_bert, my_collate_bert
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
        # sentence_ids, aspect_ids, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_heads, aspect_positions
        inputs = {'sentence': batch[0], # sentence_ids
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
        if args.pure_bert:
            # pure BERT/BERT-SPC
            inputs = {'input_ids': batch[0],
                      'input_cat_ids': batch[1],
                      'segment_ids': batch[2]}
            labels = batch[7]
        elif args.capsnet_bert:
            # CapsNet-BERT
            inputs = {'input_cat_ids': batch[1],
                      'segment_ids': batch[2]}
            labels = batch[7]
        elif args.gat_bert:
            inputs = {'input_ids': batch[0],
                      'input_aspect_ids': batch[2],
                      'word_indexer': batch[1],
                      'aspect_indexer': batch[3],
                      'input_cat_ids': batch[4],
                      'segment_ids': batch[5],
                      'dep_tags': batch[6],  # i.e. dep_tag_ids
                      'pos_class': batch[7],
                      'text_len': batch[8],
                      'aspect_len': batch[9],
                      'dep_rels': batch[11],
                      'dep_heads': batch[12],
                      'aspect_position': batch[13],
                      'dep_dirs': batch[14]}
            labels = batch[10]  # i.e. sentiment
    return inputs, labels


def get_collate_fn(args):
    embedding_type = args.embedding_type
    if embedding_type == 'glove':
        return my_collate
    elif embedding_type == 'elmo':
        return my_collate_elmo
    else:
        if args.pure_bert or args.capsnet_bert:
            return my_collate_pure_bert
        elif args.gat_bert:
            return my_collate_bert


def get_bert_optimizer(args, model, main_tag=True):
    if args.ib_lr == -1:
        # if not specified, keep it the same as args.lr
        args.ib_lr = args.lr
    if args.ib_wd == -1:
        args.ib_wd = args.weight_decay

    no_decay = ['bias', 'LayerNorm.weight']
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

    #logging.info('detected BERT params ({}): {}'.format(len(bert_name_list), bert_name_list))
    if not main_tag:
        logging.info('Learning Rate of BERT: [{}], Weight Decay of BERT: [{}]'.format(
            args.lr, args.weight_decay))
        logging.info('Learning Rate of VIB-based masking layers: [{}], Weight Decay of VIB-based masking layers: [{}]'.format(
            args.ib_lr, args.ib_wd))
        logging.info('Learning Rate of no-weight-decay params: [{}]'.format(
            args.nd_lr))
        #logging.info('detected VIB params ({}): {}'.format(len(ib_name_list), ib_name_list))
        param_groups = [{'params': ib_param_list, 'lr': args.ib_lr, 'weight_decay': args.ib_wd},
                        {'params': bert_param_list, 'lr': args.lr,
                            'weight_decay': args.weight_decay},
                        {'params': nd_param_list, 'lr': args.nd_lr, 'weight_decay': 0.0}]
    else:
        logging.info('Learning Rate of BERT: [{}], Weight Decay of BERT: [{}]'.format(
            args.lr, args.weight_decay))
        logging.info('Learning Rate of no-weight-decay params: [{}]'.format(
            args.nd_lr))
        param_groups = [{'params': bert_param_list, 'lr': args.lr, 'weight_decay': args.weight_decay},
                        {'params': nd_param_list, 'lr': args.nd_lr, 'weight_decay': 0.0}]

    if args.gat_bert:
        # default lr=1e-3, eps=1e-6, weight_decay=0.0
        return AdamW(param_groups, eps=args.adam_epsilon)
    elif args.capsnet_bert:
        return optim.Adam(param_groups, eps=args.adam_epsilon)
    elif args.pure_bert:
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train(args, train_dataset, model, test_dataset):
    '''Training the original baseline model'''
    tb_writer = SummaryWriter(comment='_single_'+str(args.source_domain))
    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    collate_fn = get_collate_fn(args)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    if args.embedding_type == 'bert':
        optimizer = get_bert_optimizer(args, model)
    else:
        parameters = filter(
            lambda param: param.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=args.lr)

    # Train
    logging.info("***** Pure Baseline Model Training *****")
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Num Epochs = %d", args.num_train_epochs)
    logging.info("  Instantaneous batch size per GPU = %d",
                 args.per_gpu_train_batch_size)
    logging.info("  Gradient Accumulation steps = %d",
                 args.gradient_accumulation_steps)
    logging.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    all_eval_results = []
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)

    # [6.7 add]
    CELoss = nn.CrossEntropyLoss()
    if args.capsnet_bert:
        CapsLoss = CapsuleLoss()
    max_acc, max_f1 = 0.0, 0.0
    for epoch in train_iterator:

        epoch_iterator = tqdm(train_dataloader, desc='Train Ep. #{}: '.format(
            epoch), total=len(train_dataloader), disable=False, ascii=True)

        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            optimizer.zero_grad()

            inputs, labels = get_input_from_batch(args, batch)
            outputs = model(**inputs)

            # note the output may contain only a single output, carefully using index!!
            if len(outputs) == 1:
                logits = outputs
            else:
                logits = outputs[0]

            # print(logits.size())
            #assert False
            if args.capsnet_bert:
                loss = CapsLoss(logits, labels)
            else:
                loss = CELoss(logits, labels)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    results, eval_loss = evaluate(args, test_dataset, model)
                    # Save model checkpoint
                    if not (args.arts_test or args.cross_domain):
                        if results['acc'] > max_acc:
                            max_acc = results['acc']
                            save_checkpoint(args, model, global_step,
                                            optimizer, main_tag=True, vib_tag=False)
                    all_eval_results.append(results)
                    for key, value in results.items():
                        tb_writer.add_scalar(
                            'eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('eval_loss', eval_loss, global_step)
                    # tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)

                    tb_writer.add_scalar(
                        'main_ce_loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            epoch_iterator.close()
            break

    tb_writer.close()
    return all_eval_results


def train_single_vib(args, main_model, train_dataset, test_dataset):
    '''Training single model with VIB'''
    tb_writer = SummaryWriter(
        comment='_single_vib_prune_'+str(args.source_domain))
    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    collate_fn = get_collate_fn(args)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    main_optimizer = get_bert_optimizer(args, main_model, main_tag=False)

    logging.info(" [VIB-based KL loss] alpha factor:{}".format(args.kl_fac))
    # Train
    logging.info(
        "***** VIB-guided Single Model Training *****")
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Num Epochs = %d", args.num_train_epochs)
    logging.info("  Instantaneous batch size per GPU = %d",
                 args.per_gpu_train_batch_size)
    logging.info("  Gradient Accumulation steps = %d",
                 args.gradient_accumulation_steps)
    logging.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_main_loss, main_logging_loss, tr_main_ce_loss, main_ce_logging_loss, tr_kl_loss, kl_logging_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    main_eval_results = []
    main_model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)

    # [6.7 add]
    CELoss = nn.CrossEntropyLoss()
    main_max_acc, main_max_f1 = 0.0, 0.0
    for epoch in train_iterator:
        main_optimizer.param_groups[0]['lr'] = args.ib_lr * \
            (args.lr_fac ** (epoch//args.lr_epoch))  # decreasing LR of VIB layers

        epoch_iterator = tqdm(train_dataloader, desc='Train Ep. #{}: '.format(
            epoch), total=len(train_dataloader), disable=False, ascii=True)

        for step, batch in enumerate(train_dataloader):
            main_model.train()

            main_optimizer.zero_grad()

            main_ce_loss, main_kl_loss = 0.0, 0.0

            batch = tuple(t.to(args.device) for t in batch)
            inputs, labels = get_input_from_batch(args, batch)
            outputs = main_model(**inputs)
            logits, _, kl_loss = outputs

            main_ce_loss = CELoss(logits, labels)
            main_kl_loss = args.kl_fac * kl_loss

            main_loss = main_ce_loss + main_kl_loss

            if args.gradient_accumulation_steps > 1:
                main_loss = main_loss / args.gradient_accumulation_steps

            main_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                main_model.parameters(), args.max_grad_norm)

            main_loss_cpu = main_loss.detach().cpu()
            main_ce_loss_cpu = main_ce_loss.detach().cpu()
            kl_loss_cpu = main_kl_loss.detach().cpu()

            tr_main_loss += main_loss_cpu.item()
            tr_main_ce_loss += main_ce_loss_cpu.item()
            tr_kl_loss += kl_loss_cpu.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # scheduler.step()  # Update learning rate schedule
                main_optimizer.step()
                main_model.zero_grad()

                global_step += 1

                # Log metrics
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Eval main network
                    results, eval_loss = evaluate(
                        args, test_dataset, main_model)
                    # Save main network
                    if not (args.arts_test or args.cross_domain):
                        if results['acc'] > main_max_acc:
                            main_max_acc = results['acc']
                            logging.info(
                                'max accuracy of pruned network: {}'.format(main_max_acc))
                            save_checkpoint(args, main_model,
                                            global_step, main_optimizer, main_tag=False, vib_tag=True)
                    main_eval_results.append(results)
                    for key, value in results.items():
                        tb_writer.add_scalar(
                            'eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar(
                        'eval_loss', eval_loss, global_step)

                    tb_writer.add_scalar(
                        'main_train_loss', (tr_main_loss - main_logging_loss) / args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'main_ce_loss', (tr_main_ce_loss - main_ce_logging_loss) / args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'kl_loss', (tr_kl_loss - kl_logging_loss) / args.logging_steps, global_step)

                    main_logging_loss = tr_main_loss
                    main_ce_logging_loss = tr_main_ce_loss
                    kl_logging_loss = tr_kl_loss

                    masks, prune_stat = main_model.get_masks()
                    logging.info(prune_stat)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            epoch_iterator.close()
            break

    tb_writer.close()
    return main_eval_results


def gatherFeatures(features):
    # Dummy vectors for allgather
    features_list = [torch.zeros_like(features)
                     for _ in range(dist.get_world_size())]
    # Allgather
    dist.all_gather(tensor_list=features_list, tensor=features.contiguous())

    # Since allgather results do not have gradients, we replace the
    # current process's corresponding embeddings with original tensors
    features_list[dist.get_rank()] = features
    features = torch.cat(features_list, 0)
    return features


def train_vib(args, main_model, sub_model, train_dataset, test_dataset):
    '''Training under VIB-guided Self-pruning Contrastive Learning Framework'''
    tb_writer = SummaryWriter(
        comment='_dual_vib_prune_'+str(args.source_domain))
    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    collate_fn = get_collate_fn(args)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    main_optimizer = get_bert_optimizer(args, main_model)
    sub_optimizer = get_bert_optimizer(args, sub_model, main_tag=False)

    logging.info(
        " [Self-pruning Contrastive Loss] alpha factor: {}, temperature: {}".format(args.sdcl_fac, args.sd_temp))
    logging.info(" [VIB-based KL loss] alpha factor:{}".format(args.kl_fac))
    # Train
    logging.info(
        "***** VIB-guided Self-pruning Contrastive Learning Framework Training *****")
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Num Epochs = %d", args.num_train_epochs)
    logging.info("  Instantaneous batch size per GPU = %d",
                 args.per_gpu_train_batch_size)
    logging.info("  Gradient Accumulation steps = %d",
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
        # do model pruning every epoch [6.7 add]
        # one shot prune
        # pruner.magnitudePruning(args.random_prune_percent)
        #logging.info('Current Network Density : {}'.format(pruner.density))

        sub_optimizer.param_groups[0]['lr'] = args.ib_lr * \
            (args.lr_fac ** (epoch//args.lr_epoch))  # decreasing LR of VIB layers

        epoch_iterator = tqdm(train_dataloader, desc='Train Ep. #{}: '.format(
            epoch), total=len(train_dataloader), disable=False, ascii=True)

        for step, batch in enumerate(train_dataloader):
            main_model.train()
            sub_model.train()

            main_optimizer.zero_grad()
            sub_optimizer.zero_grad()

            main_ce_loss, main_sdcl_loss, sub_ce_loss, kl_loss, sub_sdcl_loss = 0.0, 0.0, 0.0, 0.0, 0.0

            # 1. calculate the grad for non-pruned network
            with torch.no_grad():
                # pruned network
                batch = tuple(t.to(args.device) for t in batch)
                inputs, labels = get_input_from_batch(args, batch)
                outputs = sub_model(**inputs)
                logits = outputs[0]
                features = outputs[1]
                features_sub_no_grad = [gatherFeatures(
                    features).detach() if dist.is_initialized() else features.detach()][0]

            batch = tuple(t.to(args.device) for t in batch)
            inputs, labels = get_input_from_batch(args, batch)
            outputs = main_model(**inputs)
            logits = outputs[0]
            features = outputs[1]

            main_ce_loss = CELoss(logits, labels)
            #cl_loss = CLLoss(features_1, labels)

            features_main = [gatherFeatures(
                features) if dist.is_initialized() else features][0]
            main_sdcl_loss = SDLoss(features_sub_no_grad.unsqueeze(1),  # (bs,1,hidden_size)
                                    features_main.unsqueeze(0))  # (1,bs,hidden_size)

            main_loss = main_ce_loss + args.sdcl_fac * main_sdcl_loss

            if args.gradient_accumulation_steps > 1:
                main_loss = main_loss / args.gradient_accumulation_steps

            main_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                main_model.parameters(), args.max_grad_norm)

            main_loss_cpu = main_loss.detach().cpu()
            main_ce_loss_cpu = main_ce_loss.detach().cpu()
            main_sdcl_loss_cpu = (
                args.sdcl_fac * main_sdcl_loss).detach().cpu()

            tr_main_loss += main_loss_cpu.item()
            tr_main_ce_loss += main_ce_loss_cpu.item()
            tr_main_sdcl_loss += main_sdcl_loss_cpu.item()

            # 2. calculate the grad for pruned network
            features_main_no_grad = features_main.detach()
            batch = tuple(t.to(args.device) for t in batch)
            inputs, labels = get_input_from_batch(args, batch)
            outputs = sub_model(**inputs)
            logits = outputs[0]
            features = outputs[1]
            kl_loss = outputs[2]

            sub_ce_loss = CELoss(logits, labels)
            #cl_loss = CLLoss(features_2, labels)
            features_sub = [gatherFeatures(
                features) if dist.is_initialized() else features][0]
            sub_sdcl_loss = SDLoss(features_sub.unsqueeze(
                1), features_main_no_grad.unsqueeze(0))

            sub_loss = sub_ce_loss + args.kl_fac * kl_loss + args.sdcl_fac * sub_sdcl_loss

            if args.gradient_accumulation_steps > 1:
                sub_loss = sub_loss / args.gradient_accumulation_steps

            sub_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                sub_model.parameters(), args.max_grad_norm)

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
                    results, eval_loss = evaluate(
                        args, test_dataset, main_model)
                    # Save main network
                    if not (args.arts_test or args.cross_domain):
                        if results['acc'] > main_max_acc:
                            main_max_acc = results['acc']
                            logging.info(
                                'max accuracy of origin network: {}'.format(main_max_acc))
                            save_checkpoint(args, main_model,
                                            global_step, main_optimizer, main_tag=True, vib_tag=True)
                    main_eval_results.append(results)
                    for key, value in results.items():
                        tb_writer.add_scalar(
                            'eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar(
                        'eval_loss', eval_loss, global_step)
                    # Eval sub network
                    results, eval_loss = evaluate(
                        args, test_dataset, sub_model)
                    # Save sub network
                    if not (args.arts_test or args.cross_domain):
                        if results['acc'] > sub_max_acc:
                            sub_max_acc = results['acc']
                            logging.info(
                                'max accuracy of pruned network: {}'.format(sub_max_acc))
                            save_checkpoint(args, sub_model,
                                            global_step, sub_optimizer, main_tag=False, vib_tag=True)
                    sub_eval_results.append(results)
                    for key, value in results.items():
                        tb_writer.add_scalar(
                            'sub_eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar(
                        'sub_eval_loss', eval_loss, global_step)

                    tb_writer.add_scalar(
                        'main_train_loss', (tr_main_loss - main_logging_loss) / args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'main_ce_loss', (tr_main_ce_loss - main_ce_logging_loss) / args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'main_sdcl_loss', (tr_main_sdcl_loss - main_sdcl_logging_loss) / args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'sub_train_loss', (tr_sub_loss - sub_logging_loss) / args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'sub_ce_loss', (tr_sub_ce_loss - sub_ce_logging_loss) / args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'sub_kl_loss', (tr_sub_kl_loss - sub_kl_logging_loss) / args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'sub_sdcl_loss', (tr_sub_sdcl_loss - sub_sdcl_logging_loss) / args.logging_steps, global_step)

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


def train_sdclr(args, main_model, sub_model, train_dataset, test_dataset):
    '''Training under Self-pruning Contrastive Learning Framework'''
    tb_writer = SummaryWriter(
        comment='_dual_mag_prune_'+str(args.source_domain))
    pruner = SparsePruner(sub_model, args)
    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    collate_fn = get_collate_fn(args)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    main_optimizer = get_bert_optimizer(args, main_model)
    sub_optimizer = get_bert_optimizer(args, sub_model)

    # Train
    logging.info(
        "***** Self-pruning Contrastive Learning Framework Training *****")
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Num Epochs = %d", args.num_train_epochs)
    logging.info("  Instantaneous batch size per GPU = %d",
                 args.per_gpu_train_batch_size)
    logging.info("  Gradient Accumulation steps = %d",
                 args.gradient_accumulation_steps)
    logging.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_main_loss, main_logging_loss, tr_main_ce_loss, main_ce_logging_loss, tr_main_sdcl_loss, main_sdcl_logging_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    tr_sub_loss, sub_logging_loss, tr_sub_ce_loss, sub_ce_logging_loss, tr_sub_sdcl_loss, sub_sdcl_logging_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
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
        # do model pruning every epoch [6.7 add]
        # one shot prune
        pruner.magnitudePruning(args.random_prune_percent)
        logging.info('Current Network Density : {}'.format(pruner.density))

        epoch_iterator = tqdm(train_dataloader, desc='Train Ep. #{}: '.format(
            epoch), total=len(train_dataloader), disable=False, ascii=True)

        for step, batch in enumerate(train_dataloader):
            main_model.train()
            main_optimizer.zero_grad()

            main_ce_loss, main_sdcl_loss, sub_ce_loss, sub_sdcl_loss = 0.0, 0.0, 0.0, 0.0

            # [6.7 add]
            # 1. calculate the grad for main network
            with torch.no_grad():
                # pruned network
                batch = tuple(t.to(args.device) for t in batch)
                inputs, labels = get_input_from_batch(args, batch)
                sub_model.set_prune_flag(True)
                outputs = sub_model(**inputs)
                logits, features = outputs
                features_sub_no_grad = [gatherFeatures(
                    features).detach() if dist.is_initialized() else features.detach()][0]

            batch = tuple(t.to(args.device) for t in batch)
            inputs, labels = get_input_from_batch(args, batch)
            outputs = main_model(**inputs)

            logits, features = outputs

            main_ce_loss = CELoss(logits, labels)
            #cl_loss = CLLoss(features_1, labels)

            features_main = [gatherFeatures(
                features) if dist.is_initialized() else features][0]
            main_sdcl_loss = SDLoss(features_sub_no_grad.unsqueeze(1),
                                    features_main.unsqueeze(0))

            main_loss = main_ce_loss + args.sdcl_fac * main_sdcl_loss

            if args.gradient_accumulation_steps > 1:
                main_loss = main_loss / args.gradient_accumulation_steps

            main_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                main_model.parameters(), args.max_grad_norm)

            main_loss_cpu = main_loss.detach().cpu()
            main_ce_loss_cpu = main_ce_loss.detach().cpu()
            main_sdcl_loss_cpu = main_sdcl_loss.detach().cpu()

            tr_main_loss += main_loss_cpu.item()
            tr_main_ce_loss += main_ce_loss_cpu.item()
            tr_main_sdcl_loss += main_sdcl_loss_cpu.item()

            # 2. calculate the grad for pruned network
            features_main_no_grad = features_main.detach()
            batch = tuple(t.to(args.device) for t in batch)
            inputs, labels = get_input_from_batch(args, batch)
            sub_model.set_prune_flag(True)
            outputs = sub_model(**inputs)

            logits, features = outputs

            sub_ce_loss = CELoss(logits, labels)
            #cl_loss = CLLoss(features_2, labels)

            features_sub = [gatherFeatures(
                features) if dist.is_initialized() else features][0]
            sub_sdcl_loss = SDLoss(features_sub.unsqueeze(
                1), features_main_no_grad.unsqueeze(0))

            sub_loss = sub_ce_loss + args.sdcl_fac * sub_sdcl_loss

            if args.gradient_accumulation_steps > 1:
                sub_loss = sub_loss / args.gradient_accumulation_steps

            sub_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                sub_model.parameters(), args.max_grad_norm)

            sub_loss_cpu = sub_loss.detach().cpu()
            sub_ce_loss_cpu = sub_ce_loss.detach().cpu()
            sub_sdcl_loss_cpu = sub_sdcl_loss.detach().cpu()

            tr_sub_loss += sub_loss_cpu.item()
            tr_sub_ce_loss += sub_ce_loss_cpu.item()
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
                    results, eval_loss = evaluate(
                        args, test_dataset, main_model)
                    # Save main network
                    if not (args.arts_test or args.cross_domain):
                        if results['acc'] > main_max_acc:
                            main_max_acc = results['acc']
                            logging.info(
                                'max accuracy of origin network: {}'.format(main_max_acc))
                            save_checkpoint(args, main_model,
                                            global_step, main_optimizer, main_tag=True, vib_tag=False)
                    main_eval_results.append(results)
                    for key, value in results.items():
                        tb_writer.add_scalar(
                            'eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar(
                        'eval_loss', eval_loss, global_step)
                    # Eval sub network
                    results, eval_loss = evaluate(
                        args, test_dataset, sub_model)
                    # Save sub network
                    if not (args.arts_test or args.cross_domain):
                        if results['acc'] > sub_max_acc:
                            sub_max_acc = results['acc']
                            logging.info(
                                'max accuracy of pruned network: {}'.format(sub_max_acc))
                            save_checkpoint(args, sub_model,
                                            global_step, sub_optimizer, main_tag=False, vib_tag=False)
                    sub_eval_results.append(results)
                    for key, value in results.items():
                        tb_writer.add_scalar(
                            'sub_eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar(
                        'sub_eval_loss', eval_loss, global_step)

                    tb_writer.add_scalar(
                        'main_train_loss', (tr_main_loss - main_logging_loss) / args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'main_ce_loss', (tr_main_ce_loss - main_ce_logging_loss) / args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'main_sdcl_loss', (tr_main_sdcl_loss - main_sdcl_logging_loss) / args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'sub_train_loss', (tr_sub_loss - sub_logging_loss) / args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'sub_ce_loss', (tr_sub_ce_loss - sub_ce_logging_loss) / args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'sub_sdcl_loss', (tr_sub_sdcl_loss - sub_sdcl_logging_loss) / args.logging_steps, global_step)

                    main_logging_loss = tr_main_loss
                    main_ce_logging_loss = tr_main_ce_loss
                    main_sdcl_logging_loss = tr_main_sdcl_loss
                    sub_logging_loss = tr_sub_loss
                    sub_ce_logging_loss = tr_sub_ce_loss
                    sub_sdcl_logging_loss = tr_sub_sdcl_loss

                # Save model checkpoint?

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
    collate_fn = get_collate_fn(args)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
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
        # for batch in tqdm(eval_dataloader, desc='Evaluating'):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs, labels = get_input_from_batch(args, batch)
            outputs = model(**inputs)
            logits = outputs[0]
            # print(logits.size())
            tmp_eval_loss = CELoss(logits, labels)
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, labels.detach().cpu().numpy(), axis=0)

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


def save_checkpoint(args, model, global_step, optimizer=None, main_tag=False, vib_tag=True):
    """Saves model to checkpoints."""
    if not os.path.exists(str(args.save_folder)):
        os.mkdir(str(args.save_folder))
    save_path = str(args.save_folder)+'/checkpoints'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if args.prune:
        if vib_tag:
            save_path = save_path + '/CLR-VIB'
            masks, prune_stat = model.get_masks()
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
        file_path = os.path.join(save_path, 'origin_model.pt')
    else:
        file_path = os.path.join(save_path, 'pruned_model.pt')

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'global_step': global_step,
        'optim_state_dict': optimizer.state_dict() if optimizer is not None else '',
        'prune_stat': (prune_stat if args.prune and vib_tag else None)
    }
    torch.save(checkpoint, file_path)
    logging.info('Save model to [{}] at global training step [{}]) '.format(
        file_path, global_step))


def evaluate_badcase(args, eval_dataset, model, word_vocab):

    eval_sampler = SequentialSampler(eval_dataset)
    collate_fn = get_collate_fn(args)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=1,
                                 collate_fn=collate_fn)

    # Eval
    badcases = []
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in eval_dataloader:
        # for batch in tqdm(eval_dataloader, desc='Evaluating'):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs, labels = get_input_from_batch(args, batch)

            logits = model(**inputs)

        pred = int(np.argmax(logits.detach().cpu().numpy(), axis=1)[0])
        label = int(labels.detach().cpu().numpy()[0])
        if pred != label:
            if args.embedding_type == 'bert':
                sent_ids = inputs['input_ids'][0].detach().cpu().numpy()
                aspect_ids = inputs['input_aspect_ids'][0].detach(
                ).cpu().numpy()
                case = {}
                case['sentence'] = args.tokenizer.decode(sent_ids)
                case['aspect'] = args.tokenizer.decode(aspect_ids)
                case['pred'] = pred
                case['label'] = label
                badcases.append(case)
            else:
                sent_ids = inputs['sentence'][0].detach().cpu().numpy()
                aspect_ids = inputs['aspect'][0].detach().cpu().numpy()
                case = {}
                case['sentence'] = ' '.join(
                    [word_vocab['itos'][i] for i in sent_ids])
                case['aspect'] = ' '.join(
                    [word_vocab['itos'][i] for i in aspect_ids])
                case['pred'] = pred
                case['label'] = label
                badcases.append(case)

    return badcases


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "f1": f1
    }


def compute_metrics(preds, labels):
    return acc_and_f1(preds, labels)
