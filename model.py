from copy import deepcopy
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig
from bert_model import BertModel
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model_utils import *
from tree import *



class Aspect_Bert_GAT(nn.Module):
    '''
    RGAT-BERT + CVIB(Ours)
    '''

    def __init__(self, args, config, dep_tag_num, main_tag=True):
        '''
        main_tag: whether to add VIB-based masking layers into BERT layers, True: not use, False: use
        '''
        super(Aspect_Bert_GAT, self).__init__()
        self.args = args
        self.config = deepcopy(config)
        # VIB
        if not main_tag:
            self.config.use_ib = True
            self.kl_list = []
        else:
            self.config.use_ib = False

        self.bert = BertModel.from_pretrained(args.bert_model_dir,
                                              config=self.config,
                                              from_tf=False)

        args.embedding_dim = self.config.hidden_size  # default: 768

        # GAT
        # syntax dependency embedding
        self.dep_embed = nn.Embedding(dep_tag_num, args.embedding_dim)
        # Multi-head Relational Attention
        self.gat_dep = nn.ModuleList([
            RelationAttention(in_dim=args.embedding_dim).to(args.device)
            for i in range(args.num_heads)
        ])
        # Transformation layer
        self.dep_W = nn.Linear(args.embedding_dim * args.num_heads,
                               args.embedding_dim)
        
        self.dropout = nn.Dropout(args.dropout)
        self.gat_dropout = nn.Dropout(args.gat_dropout)

        # MLP
        last_hidden_size = args.embedding_dim * 2
        layers = [
            nn.Linear(last_hidden_size, args.final_hidden_size),
            nn.ReLU()
        ]
        for _ in range(args.num_mlps - 1):
            layers += [
                nn.Linear(args.final_hidden_size, args.final_hidden_size),
                nn.ReLU()
            ]

        # MLP layers before classifier
        self.fcs = nn.Sequential(*layers)
        # Linear classifier
        self.cls = nn.Linear(args.final_hidden_size,
                                  self.config.label_num)

    def forward(self, input_ids, input_cat_ids, segment_ids, word_indexer, input_aspect_ids,
                aspect_indexer, dep_tags, pos_class, text_len, aspect_len, dep_rels, dep_heads,
                aspect_position, dep_dirs):
        fmask = (torch.ones_like(word_indexer)
                 != word_indexer).float()  # (N，L)
        fmask[:, 0] = 1
    
        # note: different dataset has different dep_tag vocab
        dep_feature = self.dep_embed(dep_tags)

        batch_size, _ = input_cat_ids.size()

        outputs = self.bert(input_cat_ids, token_type_ids=segment_ids)
        
        feature_output = outputs[0]  # (N, L, D)
        pooler_output = outputs[1]  # (N, D)
        if self.config.use_ib:
            self.kl_list = outputs[2]
        # index select, back to original batched size.
        feature_1 = torch.stack([
            torch.index_select(f, 0, w_i)
            for f, w_i in zip(feature_output, word_indexer)
        ])
        ####################### GAT #######################
        feature_1 = self.gat_dropout(feature_1)
        dep_out_1 = [
            g(feature_1, dep_feature, fmask).unsqueeze(1)
            for g in self.gat_dep
        ]  # (N, 1, D) * num_heads
        
        dep_out_1 = torch.cat(dep_out_1, dim=1) # (N, H, D)
        dep_out_1 = self.dep_W(dep_out_1.view(batch_size, -1))  # (N, D)
        ####################### GAT #######################
        
        feature_out_1 = torch.cat([dep_out_1, pooler_output],
                                    dim=1)  # (N, 2D)
        
        final_feature_out = self.dropout(feature_out_1)
        x = self.fcs(final_feature_out)
        logits = self.cls(x)

        final_outputs = (logits, feature_out_1)
        if self.training and self.config.use_ib:
            # calculate KL loss
            kl_loss = self.kl_list[0].kld
            for ib in self.kl_list[1:]:
                kl_loss += ib.kld
            final_outputs = final_outputs + (kl_loss, )

        if self.config.output_attentions:
            final_outputs = final_outputs + (outputs[3], )

        return final_outputs

    def set_prune_flag(self, flag):
        '''set prune flag for `PrunableLinear`
        '''
        self.prune_flag = flag
        for name, module in self.bert.named_modules():
            if isinstance(module, PrunableLinear):
                if hasattr(module, "prune_mask"):
                    module.set_prune_flag(flag)

    def get_masks(self, hard_mask=True, threshold=0):
        masks = []
        if hard_mask:
            masks = [ib.get_mask_hard(threshold) for ib in self.kl_list]
            return masks, [np.sum(mask.cpu().numpy() == 0) for mask in masks]
        else:
            masks = [
                ib_layer.get_mask_weighted(threshold)
                for ib_layer in self.kl_list
            ]
            return masks


class Aspect_Bert_GCN(nn.Module):
    '''
    ASGCN + CVIB
    '''

    def __init__(self, args, config, main_tag = True):
        '''
        main_tag: whether to add VIB-based masking layers into BERT layers.
        '''
        super(Aspect_Bert_GCN, self).__init__()
        self.args = args
        self.config = deepcopy(config)
        # VIB
        if not main_tag:
            self.config.use_ib = True
            self.kl_list = []
        else:
            self.config.use_ib = False

        self.bert = BertModel.from_pretrained(args.bert_model_dir,
                                              config=self.config,
                                              from_tf=False)
        
        args.embedding_dim = self.config.hidden_size  # default: 768

        # GCN
        # gcn_hid_dim = self.args.gcn_mem_dim
        gcn_dim = args.embedding_dim 
        self.gc1 = GraphConvolution(gcn_dim, gcn_dim)
        self.gc2 = GraphConvolution(gcn_dim, gcn_dim)
        self.relu_layer = nn.ReLU()
        # self.gcn_W = nn.Linear(gcn_dim, args.embedding_dim)

        # MLP
        last_hidden_size = args.embedding_dim * 2
        layers = [
            nn.Linear(last_hidden_size, args.final_hidden_size),
            nn.ReLU()
        ]
        for _ in range(args.num_mlps - 1):
            layers += [
                nn.Linear(args.final_hidden_size, args.final_hidden_size),
                nn.ReLU()
            ]

        # MLP layers before classifier
        self.fcs = nn.Sequential(*layers)

        # Linear classifier
        self.cls = nn.Linear(args.final_hidden_size,
                                  self.config.label_num)

        self.dropout = nn.Dropout(args.dropout)
        self.gcn_dropout = nn.Dropout(args.gcn_dropout)

    def forward(self, input_cat_ids, segment_ids, text_len, aspect_len,
                left_len, adj, word_indexer):
        # text_len = torch.sum(text_indices != 0, dim=-1)
        # aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        # left_len = torch.sum(left_indices != 0, dim=-1)

        outputs = self.bert(input_cat_ids, token_type_ids=segment_ids)

        feature_output = outputs[0]  # (N, L, D)
        pooler_output = outputs[1]  # (N, D)

        # # index select, back to original batched size.
        feature_1 = torch.stack([
            torch.index_select(f, 0, w_i)
            for f, w_i in zip(feature_output, word_indexer)
        ])
    
        feature_1 = self.gcn_dropout(feature_1)
        ################# ASGCN ###################
        seq_len = feature_1.shape[1]
        adj = adj[:, :seq_len, :seq_len] # (N, L, L)
        aspect_double_idx = torch.cat(
            [left_len.unsqueeze(1), (left_len + aspect_len - 1).unsqueeze(1)],
            dim=1)

        x = self.relu_layer(
            self.gc1(
                self.position_weight(feature_1, aspect_double_idx, text_len,
                                     aspect_len), adj))
        
        x = self.relu_layer(
            self.gc2(
                self.position_weight(x, aspect_double_idx, text_len,
                                     aspect_len), adj))
       
        x = self.mask(x, aspect_double_idx)
        alpha_mat = torch.matmul(x, feature_1.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        gcn_out_1 = torch.matmul(alpha, feature_1).squeeze(1)  # (N, D)
       
        ################# ASGCN ###################

        feature_out_1 = torch.cat([gcn_out_1, pooler_output], dim=1)  # (N, 2D)

        final_feature_out = self.dropout(feature_out_1)
        x = self.fcs(final_feature_out)
        logits = self.cls(x)
        final_outputs = (logits, final_feature_out)

        if self.training and self.config.use_ib:
            # calculate KL loss
            self.kl_list = outputs[2]
            kl_loss = self.kl_list[0].kld
            for ib in self.kl_list[1:]:
                kl_loss += ib.kld
            final_outputs = final_outputs + (kl_loss, )

        if self.config.output_attentions:
            final_outputs = final_outputs + (outputs[3], )
        
        return final_outputs

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx
        text_len = text_len
        aspect_len = aspect_len
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i, 0]):
                weight[i].append(1 -
                                 (aspect_double_idx[i, 0] - j) / context_len)
            for j in range(aspect_double_idx[i, 0],
                           aspect_double_idx[i, 1] + 1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i, 1] + 1, text_len[i]):
                weight[i].append(1 -
                                 (j - aspect_double_idx[i, 1]) / context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight, dtype=torch.float).unsqueeze(2).to(
            self.args.device)
        return weight * x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i, 0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i, 0],
                           aspect_double_idx[i, 1] + 1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i, 1] + 1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask,
                            dtype=torch.float).unsqueeze(2).to(self.args.device)
        return mask * x

    def set_prune_flag(self, flag):
        '''set prune flag for `PrunableLinear`
        '''
        self.prune_flag = flag
        for name, module in self.bert.named_modules():
            if isinstance(module, PrunableLinear):
                if hasattr(module, "prune_mask"):
                    module.set_prune_flag(flag)

    def get_masks(self, hard_mask=True, threshold=0):
        masks = []
        if hard_mask:
            masks = [ib.get_mask_hard(threshold) for ib in self.kl_list]
            return masks, [np.sum(mask.cpu().numpy() == 0) for mask in masks]
        else:
            masks = [
                ib_layer.get_mask_weighted(threshold)
                for ib_layer in self.kl_list
            ]
            return masks


