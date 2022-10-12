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


class Aspect_Relational_GAT(nn.Module):
    """
    R-GAT 
    "Relational Graph Attention Network for Aspect-based Sentiment Analysis", ACL 2020
    ASGCN
    "Aspect-based Sentiment Classification with Aspect-specific Graph Convolutional Networks", EMNLP-IJCNLP 2019
    """

    def __init__(self, args, dep_tag_num):
        super(Aspect_Relational_GAT, self).__init__()
        self.args = args

        num_embeddings, embed_dim = args.glove_embedding.shape
        self.embed = nn.Embedding(num_embeddings, embed_dim)
        self.embed.weight = nn.Parameter(
            args.glove_embedding, requires_grad=False)

        self.dropout = nn.Dropout(args.dropout)
        self.tanh = nn.Tanh()

        if args.highway:
            self.highway_dep = Highway(args.num_layers, args.embedding_dim)
            self.highway = Highway(args.num_layers, args.embedding_dim)

        self.bilstm = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size,
                              bidirectional=True, batch_first=True, num_layers=args.num_layers)
        gcn_input_dim = args.hidden_size * 2

        # if args.gat:
        self.gat_dep = [RelationAttention(in_dim=args.embedding_dim).to(
            args.device) for i in range(args.num_heads)]
        if args.gat_attention_type == 'linear':
            self.gat = nn.ModuleList([LinearAttention(in_dim=gcn_input_dim, mem_dim=gcn_input_dim).to(
                args.device) for i in range(args.num_heads)])  # we prefer to keep the dimension unchanged
        elif args.gat_attention_type == 'dotprod':  # default
            self.gat = [DotprodAttention().to(args.device)
                        for i in range(args.num_heads)]
        else:
            # reshaped gcn
            self.gat = nn.Linear(gcn_input_dim, gcn_input_dim)

        self.dep_embed = nn.Embedding(dep_tag_num, args.embedding_dim)

        last_hidden_size = args.hidden_size * 4

        layers = [
            nn.Linear(last_hidden_size, args.final_hidden_size), nn.ReLU()]
        for _ in range(args.num_mlps-1):
            layers += [nn.Linear(args.final_hidden_size,
                                 args.final_hidden_size), nn.ReLU()]
        self.fcs = nn.Sequential(*layers)
        self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)

    def forward(self, sentence, aspect, pos_class, dep_tags, text_len, aspect_len, dep_rels, dep_heads, aspect_position, dep_dirs):
        '''
        Forward takes:
            sentence: sentence_id of size (batch_size, text_length)
            aspect: aspect_id of size (batch_size, aspect_length)
            pos_class: pos_tag_id of size (batch_size, text_length)
            dep_tags: dep_tag_id of size (batch_size, text_length)
            text_len: (batch_size,) length of each sentence
            aspect_len: (batch_size, ) aspect length of each sentence
            dep_rels: (batch_size, text_length) relation
            dep_heads: (batch_size, text_length) which node adjacent to that node
            aspect_position: (batch_size, text_length) mask, with the position of aspect as 1 and others as 0
            dep_dirs: (batch_size, text_length) the directions each node to the aspect
        '''
        fmask = (torch.zeros_like(sentence) != sentence).float()  # (N，L)
        dmask = (torch.zeros_like(dep_tags) != dep_tags).float()  # (N ,L)

        feature = self.embed(sentence)  # (N, L, D)
        feature = self.dropout(feature)
        aspect_feature = self.embed(aspect)  # (N, L', D)
        aspect_feature = self.dropout(aspect_feature)

        if self.args.highway:
            feature = self.highway(feature)
            aspect_feature = self.highway(aspect_feature)

        feature, _ = self.bilstm(feature)  # (N,L,D)
        aspect_feature, _ = self.bilstm(aspect_feature)  # (N,L,D)

        aspect_feature = aspect_feature.mean(dim=1)  # (N, D)

        ############################################################################################
        # do gat thing
        dep_feature = self.dep_embed(dep_tags)
        if self.args.highway:
            dep_feature = self.highway_dep(dep_feature)

        dep_out = [g(feature, dep_feature, fmask).unsqueeze(1)
                   for g in self.gat_dep]  # (N, 1, D) * num_heads
        dep_out = torch.cat(dep_out, dim=1)  # (N, H, D)
        dep_out = dep_out.mean(dim=1)  # (N, D)

        if self.args.gat_attention_type == 'gcn':
            gat_out = self.gat(feature)  # (N, L, D)
            fmask = fmask.unsqueeze(2)
            gat_out = gat_out * fmask
            gat_out = F.relu(torch.sum(gat_out, dim=1))  # (N, D)

        else:
            gat_out = [g(feature, aspect_feature, fmask).unsqueeze(1)
                       for g in self.gat]
            gat_out = torch.cat(gat_out, dim=1)
            gat_out = gat_out.mean(dim=1)

        feature_out = torch.cat([dep_out,  gat_out], dim=1)  # (N, D')
        # feature_out = gat_out
        #############################################################################################
        x = self.dropout(feature_out)
        x = self.fcs(x)
        logits = self.fc_final(x)
        return logits, feature_out


class Aspect_GAT_only(nn.Module):
    """
    GAT
    """

    def __init__(self, args, dep_tag_num, pos_tag_num):
        super(Aspect_GAT_only, self).__init__()
        self.args = args

        num_embeddings, embed_dim = args.glove_embedding.shape
        self.embed = nn.Embedding(num_embeddings, embed_dim)
        self.embed.weight = nn.Parameter(
            args.glove_embedding, requires_grad=False)

        self.dropout = nn.Dropout(args.dropout)
        self.tanh = nn.Tanh()

        if args.highway:
            self.highway = Highway(args.num_layers, args.embedding_dim)

        self.bilstm = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size,
                              bidirectional=True, batch_first=True, num_layers=args.num_layers)
        gcn_input_dim = args.hidden_size * 2

        # if args.gat:
        if args.gat_attention_type == 'linear':
            self.gat = [LinearAttention(in_dim=gcn_input_dim, mem_dim=gcn_input_dim).to(
                args.device) for i in range(args.num_heads)]  # we prefer to keep the dimension unchanged
        elif args.gat_attention_type == 'dotprod':
            self.gat = [DotprodAttention().to(args.device)
                        for i in range(args.num_heads)]
        else:
            # reshaped gcn
            self.gat = nn.Linear(gcn_input_dim, gcn_input_dim)

        last_hidden_size = args.hidden_size * 2

        layers = [
            nn.Linear(last_hidden_size, args.final_hidden_size), nn.ReLU()]
        for _ in range(args.num_mlps-1):
            layers += [nn.Linear(args.final_hidden_size,
                                 args.final_hidden_size), nn.ReLU()]
        self.fcs = nn.Sequential(*layers)
        self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)

    def forward(self, sentence, aspect, pos_class, dep_tags, text_len, aspect_len, dep_rels, dep_heads, aspect_position, dep_dirs):
        '''
        Forward takes:
            sentence: sentence_id of size (batch_size, text_length)
            aspect: aspect_id of size (batch_size, aspect_length)
            pos_class: pos_tag_id of size (batch_size, text_length)
            dep_tags: dep_tag_id of size (batch_size, text_length)
            text_len: (batch_size,) length of each sentence
            aspect_len: (batch_size, ) aspect length of each sentence
            dep_rels: (batch_size, text_length) relation
            dep_heads: (batch_size, text_length) which node adjacent to that node
            aspect_position: (batch_size, text_length) mask, with the position of aspect as 1 and others as 0
            dep_dirs: (batch_size, text_length) the directions each node to the aspect
        '''
        fmask = (torch.zeros_like(sentence) != sentence).float()  # (N，L)
        dmask = (torch.zeros_like(dep_tags) != dep_tags).float()  # (N ,L)

        feature = self.embed(sentence)  # (N, L, D)
        aspect_feature = self.embed(aspect)  # (N, L', D)
        feature = self.dropout(feature)
        aspect_feature = self.dropout(aspect_feature)

        if self.args.highway:
            feature = self.highway(feature)
            aspect_feature = self.highway(aspect_feature)

        feature, _ = self.bilstm(feature)  # (N,L,D)
        aspect_feature, _ = self.bilstm(aspect_feature)  # (N,L,D)

        aspect_feature = aspect_feature.mean(dim=1)  # (N, D)

        ############################################################################################

        if self.args.gat_attention_type == 'gcn':
            gat_out = self.gat(feature)  # (N, L, D)
            fmask = fmask.unsqueeze(2)
            gat_out = gat_out * fmask
            gat_out = F.relu(torch.sum(gat_out, dim=1))  # (N, D)

        else:
            gat_out = [g(feature, aspect_feature, fmask).unsqueeze(1)
                       for g in self.gat]
            gat_out = torch.cat(gat_out, dim=1)
            gat_out = gat_out.mean(dim=1)

        feature_out = gat_out  # (N, D')
        # feature_out = gat_out
        #############################################################################################
        x = self.dropout(feature_out)
        x = self.fcs(x)
        logit = self.fc_final(x)
        return logit


class Pure_Bert(nn.Module):
    '''
    BERT-SPC
    '''

    def __init__(self, args, config, main_tag=True):
        super(Pure_Bert, self).__init__()

        # config = BertConfig.from_pretrained(args.bert_model_dir)
        self.args = args
        self.config = deepcopy(config)
        # VIB
        if not main_tag:
            self.config.use_ib = True
            self.kl_list = []
            logging.info('Training with VIB layers.')
        else:
            self.config.use_ib = False

        self.bert = BertModel.from_pretrained(
            args.bert_model_dir, config=self.config)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        final_hidden_size = args.final_hidden_size
        # MLP for CLR
        mlp_layers = [
            nn.Linear(self.config.hidden_size, final_hidden_size), nn.ReLU()]
        for _ in range(args.num_mlps - 1):
            mlp_layers += [nn.Linear(final_hidden_size,
                                     final_hidden_size), nn.ReLU()]
        self.mlp = nn.Sequential(*mlp_layers)

        self.classifier = nn.Linear(
            self.config.hidden_size, self.config.label_num)

    def forward(self, input_ids, input_cat_ids, segment_ids):
        if self.args.spc:
            outputs = self.bert(input_cat_ids, token_type_ids=segment_ids)
        else:
            outputs = self.bert(input_ids)
        # pool output is usually *not* a good summary of the semantic content of the input,
        # you're often better with averaging or poolin the sequence of hidden-states for the whole input sequence.
        sequence_output = outputs[0]  # (N, L, D)
        pooler_output = outputs[1]  # (N, D)
        self.kl_list = outputs[2]
        # pooled_output = torch.mean(pooled_output, dim = 1)

        feature_output = self.mlp(sequence_output[:, 0])

        pooler_output = self.dropout(pooler_output)
        logits = self.classifier(pooler_output)

        if self.training and self.config.use_ib:
            # calculate KL loss
            kl_loss = self.kl_list[0].kld
            for ib in self.kl_list[1:]:
                kl_loss += ib.kld
            return logits, feature_output, kl_loss
        else:
            return logits, feature_output

    def get_masks(self, hard_mask=True, threshold=0):
        masks = []
        if hard_mask:
            masks = [ib.get_mask_hard(threshold)
                     for ib in self.kl_list]
            return masks, [np.sum(mask.cpu().numpy() == 0) for mask in masks]
        else:
            masks = [ib_layer.get_mask_weighted(
                threshold) for ib_layer in self.kl_list]
            return masks


class Aspect_Bert_GAT(nn.Module):
    '''
    RGAT-BERT & CVIB(Ours)
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
            logging.info('##### Training with VIB layers. #####')
        else:
            self.config.use_ib = False

        self.bert = BertModel.from_pretrained(
            args.bert_model_dir, config=self.config, from_tf=False)

        args.embedding_dim = self.config.hidden_size  # default: 768
        # GAT
        # syntax-dependency embedding layer
        self.dep_embed = nn.Embedding(dep_tag_num, args.embedding_dim)
        # multi-head relational attention
        self.gat_dep = nn.ModuleList([RelationAttention(in_dim=args.embedding_dim).to(
            args.device) for i in range(args.num_heads)])
        self.dep_W = nn.Linear(args.embedding_dim *
                               args.num_heads, args.embedding_dim)
        logging.info(
            '{}-heads relational self-attention of R-GAT with embedding size [{}].'.format(args.num_heads, args.embedding_dim))
        self.dropout = nn.Dropout(args.dropout)
        logging.info(
            'Dropout Rate [{}] before {}-layers MLP with hidden size [{}].'.format(args.dropout, args.num_mlps, args.final_hidden_size))
        # 3(default)-layers MLP
        last_hidden_size = args.embedding_dim * 2
        layers = [
            nn.Linear(last_hidden_size, args.final_hidden_size), nn.ReLU()]
        for _ in range(args.num_mlps - 1):
            layers += [nn.Linear(args.final_hidden_size,
                                 args.final_hidden_size), nn.ReLU()]
        # fc layers before classifier
        self.fcs = nn.Sequential(*layers)
        # linear classifier
        self.fc_final = nn.Linear(
            args.final_hidden_size, self.config.label_num)

    def forward(self, input_ids, input_aspect_ids, word_indexer, aspect_indexer, input_cat_ids, segment_ids, pos_class, dep_tags, text_len, aspect_len, dep_rels, dep_heads, aspect_position, dep_dirs):
        fmask = (torch.ones_like(word_indexer) !=
                 word_indexer).float()  # (N，L)
        fmask[:, 0] = 1
        # do gat thing
        # [8.9 add] 不同数据集的dep_tag_vocab不太一样，所以不同的dep_tag对应的id不太一样...
        dep_feature = self.dep_embed(dep_tags)

        if self.args.prune:
            batch_size, _ = input_cat_ids.size()
            if self.args.spc:
                outputs = self.bert(input_cat_ids, token_type_ids=segment_ids)
            else:
                outputs = self.bert(input_ids)
            # print(outputs.last_hidden_state.shape,batch_size,num_sent,seq_len)
            # assert False
            feature_output = outputs[0]  # (N, L, D)
            pooler_output = outputs[1]  # (N, D)
            self.kl_list = outputs[2]
            # index select, back to original batched size.
            feature_1 = torch.stack([torch.index_select(f, 0, w_i)
                                     for f, w_i in zip(feature_output, word_indexer)])
            dep_out_1 = [g(feature_1, dep_feature, fmask).unsqueeze(1)
                         for g in self.gat_dep]  # (N, 1, D) * num_heads
            '''
            # average pooling
            dep_out_1 = torch.cat(dep_out_1, dim=1)  # (N, H, D)
            dep_out_1 = dep_out_1.mean(dim=1)  # (N, D)
            '''
            # [6.8 add] replace mean pooling with concat&transform
            dep_out_1 = self.dep_W(
                torch.cat(dep_out_1, dim=1).view(batch_size, -1))  # (N, D)

            feature_out_1 = torch.cat(
                [dep_out_1, pooler_output], dim=1)  # (N, 2D)
        else:
            batch_size, _ = input_ids.size()
            if self.args.spc:
                outputs = self.bert(input_cat_ids, token_type_ids=segment_ids)
            else:
                outputs = self.bert(input_ids)
            feature_output = outputs[0]  # (N, L, D)
            pooler_output = outputs[1]  # (N, D)
            if self.config.use_ib:
                self.kl_list = outputs[2]
            # index select, back to original batched size.
            feature = torch.stack([torch.index_select(f, 0, w_i)
                                   for f, w_i in zip(feature_output, word_indexer)])

            dep_out = [g(feature, dep_feature, fmask).unsqueeze(1)
                       for g in self.gat_dep]  # (N, 1, D) * num_heads
            # average pooling
            dep_out = torch.cat(dep_out, dim=1)  # (N, H, D)
            dep_out = dep_out.mean(dim=1)  # (N, D)
            '''
            # linear transform
            # [6.8 add] replace mean pooling with concat&transform
            dep_out = self.dep_W(torch.cat(dep_out, dim=1).view(
                batch_size, -1))  # (N, D)
            '''
            feature_out_1 = torch.cat(
                [dep_out, pooler_output], dim=1)  # (N, 2D)

        #############################################################################################
        final_feature_out = self.dropout(feature_out_1)
        x = self.fcs(final_feature_out)
        logits = self.fc_final(x)
        final_outputs = (logits, feature_out_1)
        if self.training and self.config.use_ib:
            # calculate KL loss
            kl_loss = self.kl_list[0].kld
            for ib in self.kl_list[1:]:
                kl_loss += ib.kld
            final_outputs = final_outputs + (kl_loss,)

        if self.config.output_attentions:
            final_outputs = final_outputs + (outputs[3],)

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
            masks = [ib.get_mask_hard(threshold)
                     for ib in self.kl_list]
            return masks, [np.sum(mask.cpu().numpy() == 0) for mask in masks]
        else:
            masks = [ib_layer.get_mask_weighted(
                threshold) for ib_layer in self.kl_list]
            return masks


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0


class CapsNet_Bert(nn.Module):
    '''
    CapsNet-BERT
     "A Challenge Dataset and Effective Models for Aspect-Based Sentiment Analysis", EMNLP-IJCNLP 2019
    '''

    def __init__(self, args, config, main_tag=True, dropout=0.1):
        super(CapsNet_Bert, self).__init__()
        self.bert = BertModel.from_pretrained(
            args.bert_model_dir, config=config, from_tf=False)
        self.config = deepcopy(config)
        # VIB
        if not main_tag:
            self.config.use_ib = True
            self.kl_list = []
            logging.info('Training with VIB layers.')
        else:
            self.config.use_ib = False
        num_categories = self.config.label_num
        capsule_size = 300
        bert_size = self.config.hidden_size  # default 768
        self.aspect_transform = nn.Sequential(
            nn.Linear(bert_size, capsule_size),
            nn.Dropout(dropout)
        )
        self.sentence_transform = nn.Sequential(
            nn.Linear(bert_size, capsule_size),
            nn.Dropout(dropout)
        )
        self.norm_attention = BilinearAttention(capsule_size, capsule_size)
        self.guide_capsule = nn.Parameter(
            torch.Tensor(num_categories, capsule_size)
        )
        self.guide_weight = nn.Parameter(
            torch.Tensor(capsule_size, capsule_size)
        )
        self.scale = nn.Parameter(torch.tensor(5.0))
        self.capsule_projection = nn.Linear(
            bert_size, bert_size * num_categories)
        # output-feature-for-CLR dropout
        self.feature_dropout = nn.Dropout(args.dropout)
        self.dropout = dropout
        self.num_categories = num_categories
        self._reset_parameters()
        self._load_sentiment('/data1/mschang/sentiment_matrix.npy')

    def _reset_parameters(self):
        init.xavier_uniform_(self.guide_capsule)
        init.xavier_uniform_(self.guide_weight)

    def _load_sentiment(self, path):
        sentiment = np.load(path)
        e1 = np.mean(sentiment)
        d1 = np.std(sentiment)
        e2 = 0
        d2 = np.sqrt(2.0 / (sentiment.shape[0] + sentiment.shape[1]))
        sentiment = (sentiment - e1) / d1 * d2 + e2
        self.guide_capsule.data.copy_(torch.tensor(sentiment))

    def forward(self, input_cat_ids, segment_ids):
        # BERT encoding

        outputs = self.bert(input_ids=input_cat_ids,
                            token_type_ids=segment_ids)
        encoder_layer = outputs[0]  # (N, L, D)
        self.kl_list = outputs[2]

        batch_size, segment_len = segment_ids.size()
        max_segment_len = segment_ids.argmax(dim=-1, keepdim=True)
        batch_arrange = torch.arange(segment_len).unsqueeze(
            0).expand(batch_size, segment_len).to(segment_ids.device)
        segment_mask = batch_arrange <= max_segment_len
        sentence_mask = segment_mask & (1 - segment_ids).byte()
        aspect_mask = segment_ids
        sentence_lens = sentence_mask.long().sum(dim=1, keepdim=True)
        # aspect average pooling
        aspect_lens = aspect_mask.long().sum(dim=1, keepdim=True)
        aspect = encoder_layer.masked_fill(aspect_mask.unsqueeze(-1) == 0, 0)
        aspect = aspect.sum(dim=1, keepdim=False) / aspect_lens.float()
        # sentence encode layer
        max_len = sentence_lens.max().item()
        sentence = encoder_layer[:, 0: max_len].contiguous()
        sentence_mask = sentence_mask[:, 0: max_len].contiguous()
        sentence = sentence.masked_fill(sentence_mask.unsqueeze(-1) == 0, 0)
        # primary capsule layer
        sentence = self.sentence_transform(sentence)
        primary_capsule = squash(sentence, dim=-1)  # (N,L,C)
        # print(primary_capsule.size())
        feature_output = self.feature_dropout(
            primary_capsule[:, 0])  # [9.1 add] feature_output for CLR
        aspect = self.aspect_transform(aspect)
        aspect_capsule = squash(aspect, dim=-1)
        # aspect aware normalization
        norm_weight = self.norm_attention.get_attention_weights(
            aspect_capsule, primary_capsule, sentence_mask)
        # capsule guided routing
        category_capsule = self._capsule_guided_routing(
            primary_capsule, norm_weight)
        category_capsule_norm = torch.sqrt(
            torch.sum(category_capsule * category_capsule, dim=-1, keepdim=False))
        # print(category_capsule_norm.size())
        if self.training and self.config.use_ib:
            # calculate KL loss
            kl_loss = self.kl_list[0].kld
            for ib in self.kl_list[1:]:
                kl_loss += ib.kld
            return category_capsule_norm, feature_output, kl_loss
        else:
            return category_capsule_norm, feature_output  # (N, 3)

    def _capsule_guided_routing(self, primary_capsule, norm_weight):
        guide_capsule = squash(self.guide_capsule)
        guide_matrix = primary_capsule.matmul(
            self.guide_weight).matmul(guide_capsule.transpose(0, 1))
        guide_matrix = F.softmax(guide_matrix, dim=-1)
        # (batch_size, time_step, num_categories)
        guide_matrix = guide_matrix * norm_weight.unsqueeze(-1) * self.scale
        category_capsule = guide_matrix.transpose(1, 2).matmul(primary_capsule)
        category_capsule = F.dropout(
            category_capsule, p=self.dropout, training=self.training)
        category_capsule = squash(category_capsule)
        # print(category_capsule.size())
        return category_capsule

    def get_masks(self, hard_mask=True, threshold=0):
        masks = []
        if hard_mask:
            masks = [ib.get_mask_hard(threshold) for ib in self.kl_list]
            return masks, [np.sum(mask.cpu().numpy() == 0) for mask in masks]
        else:
            masks = [ib_layer.get_mask_weighted(
                threshold) for ib_layer in self.kl_list]
            return masks


class ATAE_LSTM(nn.Module):
    '''
    ATAE-LSTM
    "Attention-based LSTM for Aspect-level Sentiment Classification", EMNLP 2016
    '''

    def __init__(self, args):
        super(ATAE_LSTM, self).__init__()
        self.args = args
        num_embeddings, embed_dim = args.glove_embedding.shape
        self.embed = nn.Embedding(num_embeddings, embed_dim)
        self.embed.weight = nn.Parameter(
            args.glove_embedding, requires_grad=False)
        #self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding()
        self.lstm = DynamicLSTM(
            embed_dim*2, args.hidden_dim, num_layers=1, batch_first=True)
        self.attention = NoQueryAttention(
            args.hidden_dim+embed_dim, score_function='bi_linear')
        self.dense = nn.Linear(args.hidden_dim, args.num_classes)
        self.dense2 = nn.Linear(args.hidden_dim, 2)

    def forward(self, text_indices, aspect_indices):
        #text_indices, aspect_indices = inputs[0], inputs[1]
        x_len = torch.sum(text_indices != 0, dim=-1)
        x_len_max = torch.max(x_len)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1).float()

        x = self.embed(text_indices)
        x = self.squeeze_embedding(x, x_len)

        aspect = self.embed(aspect_indices)
        aspect_pool = torch.div(torch.sum(aspect, dim=1),
                                aspect_len.unsqueeze(1))
        aspect = aspect_pool.unsqueeze(1).expand(-1, x_len_max, -1)
        x = torch.cat((aspect, x), dim=-1)

        h, (_, _) = self.lstm(x, x_len)
        ha = torch.cat((h, aspect), dim=-1)
        _, score = self.attention(ha)
        output = torch.squeeze(torch.bmm(score, h), dim=1)

        out = self.dense(output)
        out2 = self.dense2(output)
        return out, output, out2


class MemNet(nn.Module):
    '''
    MemNet
    "Aspect Level Sentiment Classification with Deep Memory Network", EMNLP 2016
    '''

    def locationed_memory(self, memory, memory_len):
        # here we just simply calculate the location vector in Model2's manner
        batch_size = memory.shape[0]
        seq_len = memory.shape[1]
        memory_len = memory_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for idx in range(memory_len[i]):
                weight[i].append(1-float(idx+1)/memory_len[i])
            for idx in range(memory_len[i], seq_len):
                weight[i].append(1)
        weight = torch.tensor(weight).to(self.args.device)
        memory = weight.unsqueeze(2)*memory
        return memory

    def __init__(self, args):
        super(MemNet, self).__init__()
        self.args = args
        num_embeddings, embed_dim = args.glove_embedding.shape
        self.embed = nn.Embedding(num_embeddings, embed_dim)
        self.embed.weight = nn.Parameter(
            args.glove_embedding, requires_grad=False)
        self.squeeze_embedding = SqueezeEmbedding(batch_first=True)
        self.attention = Attention(embed_dim, score_function='mlp')
        self.x_linear = nn.Linear(embed_dim, embed_dim)
        self.dense = nn.Linear(embed_dim, args.num_classes)

    def forward(self, inputs):
        text_raw_without_aspect_indices, aspect_indices = inputs[0], inputs[1]
        memory_len = torch.sum(text_raw_without_aspect_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        nonzeros_aspect = torch.tensor(
            aspect_len, dtype=torch.float).to(self.args.device)

        memory = self.embed(text_raw_without_aspect_indices)
        memory = self.squeeze_embedding(memory, memory_len)
        # memory = self.locationed_memory(memory, memory_len)
        aspect = self.embed(aspect_indices)
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, nonzeros_aspect.view(
            nonzeros_aspect.size(0), 1))
        x = aspect.unsqueeze(dim=1)
        for _ in range(self.args.hops):
            x = self.x_linear(x)
            out_at, _ = self.attention(memory, x)
            x = out_at + x
        x = x.view(x.size(0), -1)
        out = self.dense(x)
        return out


class IAN(nn.Module):
    '''
    IAN
    "Interactive Attention Networks for Aspect-Level Sentiment Classification", IJCAI 2017
    '''

    def __init__(self, args):
        super(IAN, self).__init__()
        self.args = args
        num_embeddings, embed_dim = args.glove_embedding.shape
        self.embed = nn.Embedding(num_embeddings, embed_dim)
        self.embed.weight = nn.Parameter(
            args.glove_embedding, requires_grad=False)

        self.lstm_context = DynamicLSTM(
            embed_dim, args.hidden_dim, num_layers=1, batch_first=True)
        self.lstm_aspect = DynamicLSTM(
            embed_dim, args.hidden_dim, num_layers=1, batch_first=True)
        self.attention_aspect = Attention(
            args.hidden_dim, score_function='bi_linear')
        self.attention_context = Attention(
            args.hidden_dim, score_function='bi_linear')
        self.dense = nn.Linear(args.hidden_dim*2, args.num_classes)

    def forward(self, inputs):
        text_raw_indices, aspect_indices = inputs[0], inputs[1]
        text_raw_len = torch.sum(text_raw_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)

        context = self.embed(text_raw_indices)
        aspect = self.embed(aspect_indices)
        context, (_, _) = self.lstm_context(context, text_raw_len)
        aspect, (_, _) = self.lstm_aspect(aspect, aspect_len)

        aspect_len = torch.tensor(
            aspect_len, dtype=torch.float).to(self.args.device)
        aspect_pool = torch.sum(aspect, dim=1)
        aspect_pool = torch.div(
            aspect_pool, aspect_len.view(aspect_len.size(0), 1))

        text_raw_len = torch.tensor(
            text_raw_len, dtype=torch.float).to(self.args.device)
        context_pool = torch.sum(context, dim=1)
        context_pool = torch.div(
            context_pool, text_raw_len.view(text_raw_len.size(0), 1))

        aspect_final, _ = self.attention_aspect(aspect, context_pool)
        aspect_final = aspect_final.squeeze(dim=1)
        context_final, _ = self.attention_context(context, aspect_pool)
        context_final = context_final.squeeze(dim=1)

        x = torch.cat((aspect_final, context_final), dim=-1)
        out = self.dense(x)
        return out, x
