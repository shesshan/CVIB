from typing import List, Tuple, Optional, overload
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch import _VF
from torch.nn.utils.rnn import PackedSequence
import math
from copy import deepcopy
from collections import OrderedDict
import numpy as np

PAD = '<pad>'
UNK = '<unk>'
ASPECT = '<aspect>'

PAD_INDEX = 0
UNK_INDEX = 1
ASPECT_INDEX = 2

INF = 1e9

############### VIB masking #################


def reparameterize(mu, logvar, batch_size, cuda=False, sampling=True):
    # output: (batch_size, dim)
    if sampling:
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(batch_size, std.size(0)).cuda(
            mu.get_device()).normal_()
        eps = Variable(eps)
        return mu.view(1, -1) + eps * std.view(1, -1)
    else:
        return mu.view(1, -1)


class InformationBottleneck(nn.Module):
    '''
    VIB-based masking layer
    '''

    def __init__(self, dim, mask_thresh=0, init_mag=9, init_var=0.01,
                 kl_mult=1, divide_w=False, sample_in_training=True, sample_in_testing=False, masking=False):
        super(InformationBottleneck, self).__init__()
        # [7.21 add] dim means out_channels
        self.prior_z_logD = Parameter(torch.Tensor(dim))
        self.post_z_mu = Parameter(torch.Tensor(dim))
        self.post_z_logD = Parameter(torch.Tensor(dim))

        self.epsilon = 1e-8
        self.dim = dim
        self.sample_in_training = sample_in_training
        self.sample_in_testing = sample_in_testing
        # if masking=True, apply mask directly
        self.masking = masking

        # initialization
        stdv = 1. / math.sqrt(dim)
        self.post_z_mu.data.normal_(1, init_var)
        self.prior_z_logD.data.normal_(-init_mag, init_var)
        self.post_z_logD.data.normal_(-init_mag, init_var)

        self.need_update_z = True  # flag for updating z during testing
        self.mask_thresh = mask_thresh
        self.kl_mult = kl_mult
        self.divide_w = divide_w
        self.kld = None

    def adapt_shape(self, src_shape, x_shape):
        # to distinguish bert hidden layers and fc layers
        # see if we need to expand the dimension of z
        # (bs,1,dim) for bert layers
        # (bs,dim) for fc layers
        new_shape = src_shape if len(src_shape) == 2 else (
            1, src_shape[0])  # (bs/1, dim)
        if len(x_shape) > 2:
            ori_shape = list(new_shape)
            new_shape = [ori_shape[0], 1, ori_shape[1]]  # (bs/1, 1, dim)
        return new_shape

    def get_logalpha(self):
        # if (self.post_z_mu.data.pow(2) + self.epsilon) > 0.0:
        #   print('hello')
        return self.post_z_logD.data - torch.log(self.post_z_mu.data.pow(2) + self.epsilon)

    def get_dp(self):
        logalpha = self.get_logalpha()
        alpha = torch.exp(logalpha)
        return alpha / (1+alpha)

    def get_mask_hard(self, threshold=0):
        logalpha = self.get_logalpha()
        hard_mask = (logalpha < threshold).float()
        return hard_mask

    def get_mask_weighted(self, threshold=0):
        logalpha = self.get_logalpha()
        mask = (logalpha < threshold).float() * self.post_z_mu.data
        return mask

    def set_testing_mode(self):
        self.sample_in_testing = True

    def forward(self, x):
        # 4 modes: sampling, hard mask, weighted mask, use mean value
        if self.masking:
            mask = self.get_mask_hard(self.mask_thresh)
            new_shape = self.adapt_shape(mask.size(), x.size())
            return x * Variable(mask.view(new_shape))

        bsize = x.size(0)
        if (self.training and self.sample_in_training) or (not self.training and self.sample_in_testing):
            #logging.info('Reparameterizing & Masking in training process.')
            z_scale = reparameterize(
                self.post_z_mu, self.post_z_logD, bsize, cuda=True, sampling=True)  # (bs, dim)
            if not self.training:
                z_scale *= Variable(self.get_mask_hard(self.mask_thresh))
        else:
            #logging.info('Masking in testing process.')
            z_scale = Variable(self.get_mask_weighted(self.mask_thresh))

        self.kld = self.kl_closed_form(x)
        new_shape = self.adapt_shape(
            z_scale.size(), x.size())  # [bs, [1], dim]
        return x * z_scale.view(new_shape)  # (bs, [seq_len], dim)

    def kl_closed_form(self, x):
        h_D = torch.exp(self.post_z_logD)
        h_mu = self.post_z_mu
        KLD = torch.sum(torch.log(1 + h_mu.pow(2) / (h_D + self.epsilon)))
        KLD *= x.size()[1]  # multiply sequence length
        return KLD * 0.5 * self.kl_mult


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)


def squash(x, dim=-1):
    squared = torch.sum(x * x, dim=dim, keepdim=True)
    scale = torch.sqrt(squared) / (1.0 + squared)
    return scale * x


class Attention(nn.Module):
    """
    The base class of attention.
    """

    def __init__(self, dropout):
        super(Attention, self).__init__()
        self.dropout = dropout

    def forward(self, query, key, value, mask=None):
        """
        query: FloatTensor (batch_size, query_size) or FloatTensor (batch_size, num_queries, query_size)
        key: FloatTensor (batch_size, time_step, key_size)
        value: FloatTensor (batch_size, time_step, hidden_size)
        mask: ByteTensor (batch_size, time_step) or ByteTensor (batch_size, num_queries, time_step)
        """
        single_query = False
        if len(query.size()) == 2:
            query = query.unsqueeze(1)
            single_query = True
        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(1)
            else:
                assert mask.size(1) == query.size(1)
        # FloatTensor (batch_size, num_queries, time_step)
        score = self._score(query, key)
        weights = self._weights_normalize(score, mask)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        output = weights.matmul(value)
        if single_query:
            output = output.squeeze(1)
        return output

    def _score(self, query, key):
        raise NotImplementedError('Attention score method is not implemented.')

    def _weights_normalize(self, score, mask):
        if not mask is None:
            score = score.masked_fill(mask == 0, -INF)
        weights = F.softmax(score, dim=-1)
        return weights

    def get_attention_weights(self, query, key, mask=None):
        single_query = False
        if len(query.size()) == 2:
            query = query.unsqueeze(1)
            single_query = True
        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(1)
            else:
                assert mask.size(1) == query.size(1)
        # FloatTensor (batch_size, num_queries, time_step)
        score = self._score(query, key)
        weights = self._weights_normalize(score, mask)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        if single_query:
            weights = weights.squeeze(1)
        return weights


class BilinearAttention(Attention):

    def __init__(self, query_size, key_size, dropout=0):
        super(BilinearAttention, self).__init__(dropout)
        self.weights = nn.Parameter(torch.FloatTensor(query_size, key_size))
        init.xavier_uniform_(self.weights)

    def _score(self, query, key):
        """
        query: FloatTensor (batch_size, num_queries, query_size)
        key: FloatTensor (batch_size, time_step, key_size)
        """
        score = query.matmul(self.weights).matmul(key.transpose(1, 2))
        return score


class Attention_2(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention_2, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous(
        ).view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous(
        ).view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            # (n_head*?, q_len, k_len, hidden_dim*2)
            kq = torch.cat((kxx, qxx), dim=-1)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            score = F.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1)
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        # (?, q_len, n_head*hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output, score


class NoQueryAttention(Attention_2):
    '''q is a parameter'''

    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', q_len=1, dropout=0):
        super(NoQueryAttention, self).__init__(
            embed_dim, hidden_dim, out_dim, n_head, score_function, dropout)
        self.q_len = q_len
        self.q = nn.Parameter(torch.Tensor(q_len, embed_dim))
        self.reset_q()

    def reset_q(self):
        stdv = 1. / math.sqrt(self.embed_dim)
        self.q.data.uniform_(-stdv, stdv)

    def forward(self, k, **kwargs):
        mb_size = k.shape[0]
        q = self.q.expand(mb_size, -1, -1)
        return super(NoQueryAttention, self).forward(k, q)


class RelationAttention(nn.Module):
    def __init__(self, in_dim=300, hidden_dim=64):
        # in_dim: the dimension fo query vector
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, feature, dep_tags_v, dmask):
        '''
        C feature/context [N, L, D]
        Q dep_tags_v          [N, L, D]
        mask dmask          [N, L]
        '''
        Q = self.fc1(dep_tags_v)
        Q = self.relu(Q)
        Q = self.fc2(Q)  # (N, L, 1)
        Q = Q.squeeze(2)
        Q = F.softmax(mask_logits(Q, dmask), dim=1)

        Q = Q.unsqueeze(2)
        out = torch.bmm(feature.transpose(1, 2), Q)
        out = out.squeeze(2)
        # out = F.sigmoid(out)
        return out  # ([N, L])


class LinearAttention(nn.Module):
    '''
    re-implement of gat's attention
    '''

    def __init__(self, in_dim=300, mem_dim=300):
        # in dim, the dimension of query vector
        super().__init__()
        self.linear = nn.Linear(in_dim, mem_dim)
        self.fc = nn.Linear(mem_dim * 2, 1)
        self.leakyrelu = nn.LeakyReLU(1e-2)

    def forward(self, feature, aspect_v, dmask):
        '''
        C feature/context [N, L, D]
        Q dep_tags_v          [N, D]
        mask dmask          [N, L]
        '''

        Q = self.linear(aspect_v)  # (N, D)
        Q = Q.unsqueeze(1)  # (N, 1, D)
        Q = Q.expand_as(feature)  # (N, L, D)
        Q = self.linear(Q)  # (N, L, D)
        feature = self.linear(feature)  # (N, L, D)

        att_feature = torch.cat([feature, Q], dim=2)  # (N, L, 2D)
        att_weight = self.fc(att_feature)  # (N, L, 1)
        dmask = dmask.unsqueeze(2)  # (N, L, 1)
        att_weight = mask_logits(att_weight, dmask)  # (N, L ,1)

        attention = F.softmax(att_weight, dim=1)  # (N, L, 1)

        out = torch.bmm(feature.transpose(1, 2), attention)  # (N, D, 1)
        out = out.squeeze(2)
        # out = F.sigmoid(out)

        return out


class DotprodAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feature, aspect_v, dmask):
        '''
        C feature/context [N, L, D]
        Q dep_tags_v          [N, D]
        mask dmask          [N, L]
        '''

        Q = aspect_v
        Q = Q.unsqueeze(2)  # (N, D, 1)
        dot_prod = torch.bmm(feature, Q)  # (N, L, 1)
        dmask = dmask.unsqueeze(2)  # (N, D, 1)
        attention_weight = mask_logits(dot_prod, dmask)  # (N, L ,1)
        attention = F.softmax(attention_weight, dim=1)  # (N, L, 1)

        out = torch.bmm(feature.transpose(1, 2), attention)  # (N, D, 1)
        out = out.squeeze(2)
        # out = F.sigmoid(out)
        # (N, D), ([N, L]), (N, L, 1)
        return out


class Highway(nn.Module):
    def __init__(self, layer_num, dim):
        super().__init__()
        self.layer_num = layer_num
        self.linear = nn.ModuleList([nn.Linear(dim, dim)
                                     for _ in range(layer_num)])
        self.gate = nn.ModuleList([nn.Linear(dim, dim)
                                   for _ in range(layer_num)])

    def forward(self, x):
        for i in range(self.layer_num):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = F.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        return x


class DepparseMultiHeadAttention(nn.Module):
    def __init__(self, h=6, Co=300, cat=True):
        super().__init__()
        self.hidden_size = Co // h
        self.h = h
        self.fc1 = nn.Linear(Co, Co)
        self.relu = nn.ReLU()
        self.fc2s = nn.ModuleList(
            [nn.Linear(self.hidden_size, 1) for _ in range(h)])
        self.cat = cat

    def forward(self, feature, dep_tags_v, dmask):
        '''
        C feature/context [N, L, D]
        Q dep_tags_v          [N, L, D]
        mask dmask          [N, L]
        '''
        nbatches = dep_tags_v.size(0)
        Q = self.fc1(dep_tags_v).view(nbatches, -1, self.h,
                                      self.hidden_size)  # [N, L, #heads, hidden_size]
        Q = self.relu(Q)
        Q = Q.transpose(0, 2)  # [#heads, L, N, hidden_size]
        Q = [l(q).squeeze(2).transpose(0, 1)
             for l, q in zip(self.fc2s, Q)]  # [N, L] * #heads
        # Q = Q.squeeze(2)
        Q = [F.softmax(mask_logits(q, dmask), dim=1).unsqueeze(2)
             for q in Q]  # [N, L, 1] * #heads

        # Q = Q.unsqueeze(2)
        if self.cat:
            out = torch.cat(
                [torch.bmm(feature.transpose(1, 2), q).squeeze(2) for q in Q], dim=1)
        else:
            out = torch.stack(
                [torch.bmm(feature.transpose(1, 2), q).squeeze(2) for q in Q], dim=2)
            out = torch.sum(out, dim=2)
        # out = out.squeeze(2)
        return out, Q[0]  # ([N, L]) one head

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_hid, d_inner_hid=None, dropout=0):
        super(PositionwiseFeedForward, self).__init__()
        if d_inner_hid is None:
            d_inner_hid = d_hid
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_inner_hid, d_hid, 1)  # position-wise
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        return output

class SqueezeEmbedding(nn.Module):
    """
    Squeeze sequence embedding length to the longest one in the batch
    """
    def __init__(self, batch_first=True):
        super(SqueezeEmbedding, self).__init__()
        self.batch_first = batch_first

    def forward(self, x, x_len):
        """
        sequence -> sort -> pad and pack -> unpack ->unsort
        :param x: sequence embedding vectors
        :param x_len: numpy/tensor list
        :return:
        """
        """sort"""
        x_sort_idx = torch.sort(-x_len)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len.cpu(), batch_first=self.batch_first)
        """unpack: out"""
        out = torch.nn.utils.rnn.pad_packed_sequence(x_emb_p, batch_first=self.batch_first)  # (sequence, lengths)
        out = out[0]  #
        """unsort"""
        out = out[x_unsort_idx]
        return out


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config, dual_size=False):
        super().__init__()
        hidden_size = config.hidden_size
        if dual_size:
            self.dense = nn.Linear(2*hidden_size, hidden_size)
        else:
            self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding.
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2",
                                    "avg_first_last", "max_avg"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, last_hidden, weighted=False, norm_weights=None):
        '''
        last_hidden (bs * sent_num, len, hidden_size)
        attention_mask (bs * sent_num, len)
        norm_weights (if not None) (bs * sent_num ,len)

        return: pooling result (bs,hidden_size)
        '''

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type in ['avg']:
            if weighted:
                if len(norm_weights.size()) == 2:
                    norm_weights = norm_weights.unsqueeze(-1)  # (bs,len,1)
                last_hidden = last_hidden * \
                    norm_weights  # (bs,len,hid)
            # (bs * sent_num, hidden_size) / (bs * sent_num, 1)
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type in ['max_avg']:
            if weighted:
                if len(norm_weights.size()) == 2:
                    norm_weights = norm_weights.unsqueeze(-1)  # (bs,len,1)
                last_hidden = last_hidden * norm_weights  # (bs,len,hid)
            return last_hidden.max(dim=1).values
        else:
            raise NotImplementedError


################ Magnitude Pruning ################
class SparsePruner(object):
    """Performs pruning on the given model."""

    def __init__(self, model, args, reset=True):
        super(SparsePruner, self).__init__()
        self.model = model
        self.args = args
        if reset:
            self.reset()

    def magnitudePruning(self, randomPruneFraction=0.0):
        prunePercent = self.args.prune_percent
        randomPruneFraction = self.args.random_prune_percent
        magnitudePruneFraction = prunePercent - randomPruneFraction
        weights = []
        for name, module in self.model.bert.named_modules():
            if isinstance(module, PrunableLinear):
                if hasattr(module, "prune_mask"):
                    weights.append(
                        module.weight.clone().cpu().detach().numpy())

        # recover all prune_mask from last pruning
        self.reset()
        prunableTensors = []
        for name, module in self.model.bert.named_modules():
            if isinstance(module, PrunableLinear):
                if hasattr(module, "prune_mask"):
                    prunableTensors.append(module.prune_mask.detach())
        number_of_remaining_weights = torch.sum(torch.tensor(
            [torch.sum(v) for v in prunableTensors])).cpu().numpy()
        number_of_weights_to_prune_magnitude = np.ceil(
            magnitudePruneFraction * number_of_remaining_weights).astype(int)
        number_of_weights_to_prune_random = np.ceil(
            randomPruneFraction * number_of_remaining_weights).astype(int)
        random_prune_prob = number_of_weights_to_prune_random / \
            (number_of_remaining_weights - number_of_weights_to_prune_magnitude)

        # Create a vector of all the unpruned weights in the model.
        weight_vector = np.concatenate([v.flatten() for v in weights])
        threshold = np.sort(np.abs(weight_vector))[min(
            number_of_weights_to_prune_magnitude, len(weight_vector) - 1)]

        # apply the mask
        for name, module in self.model.bert.named_modules():
            if isinstance(module, PrunableLinear):
                if hasattr(module, "prune_mask"):
                    module.prune_mask = (
                        torch.abs(module.weight) >= threshold).float()
                    # random weights been pruned
                    module.prune_mask[torch.rand_like(
                        module.prune_mask) < random_prune_prob] = 0

    def reset(self):
        for name, module in self.model.bert.named_modules():
            if isinstance(module, PrunableLinear):
                if hasattr(module, "prune_mask"):
                    module.prune_mask = torch.ones_like(module.weight)

    @property
    def sparsity(self):
        """Return the percent of weights that have been pruned as a decimal."""
        prunableTensors = []
        for name, module in self.model.bert.named_modules():
            if isinstance(module, PrunableLinear):
                if hasattr(module, "prune_mask"):
                    prunableTensors.append(module.prune_mask.detach())

        unpruned = torch.sum(torch.tensor(
            [torch.sum(v) for v in prunableTensors]))
        total = torch.sum(torch.tensor(
            [torch.sum(torch.ones_like(v)) for v in prunableTensors]))
        return 1 - unpruned.float() / total.float()

    @property
    def density(self):
        return 1 - self.sparsity

    def save_mask(self, epoch, filename):
        pruneMask = OrderedDict()
        for name, module in self.model.bert.named_modules():
            if isinstance(module, PrunableLinear):
                if hasattr(module, "prune_mask"):
                    pruneMask[name] = module.prune_mask.cpu().type(torch.bool)

        torch.save({"epoch": epoch, "pruneMask": pruneMask}, filename)

    def load_mask(self, state_dict):
        pruneMask = state_dict["pruneMask"]
        for name, module in self.model.bert.named_modules():
            if isinstance(module, PrunableLinear):
                if hasattr(module, "prune_mask"):
                    module.prune_mask.data = pruneMask[name].to(
                        module.weight.data.device).float()


class PrunableLinear(nn.Module):
    '''
    make linear layer (of BERT) prunable
    '''

    __constants__ = ['bias', 'in_features',
                     'out_features', 'prune_flag', 'prune_mask']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        # NOTE:customerly adding prune_mask & prune_flag
        self.prune_mask = torch.ones(list(self.weight.shape))
        self.prune_flag = False

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        if self.prune_flag:
            weight = self.weight * self.prune_mask
        else:
            weight = self.weight
        return F.linear(input, weight, self.bias)

    def set_prune_flag(self, flag):
        self.prune_flag = flag

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class DRGRU(nn.RNNBase):
    def __init__(self, config, *args, **kwargs):
        super(DRGRU, self).__init__('GRU', *args, **kwargs)
        self.config = config
        if self.config.norm_gru:
            self._reset_parameters()

    @overload
    @torch._jit_internal._overload_method  # noqa: F811
    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:  # noqa: F811
        pass

    @overload
    @torch._jit_internal._overload_method  # noqa: F811
    def forward(self, input: PackedSequence, hx: Optional[Tensor] = None) -> Tuple[PackedSequence, Tensor]:  # noqa: F811
        pass

    def forward(self, input, hx=None):  # noqa: F811
        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = input.size(
                0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.zeros(self.num_layers * num_directions,
                             max_batch_size, self.hidden_size,
                             dtype=input.dtype, device=input.device)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        if batch_sizes is None:
            result = _VF.gru(input, hx, self._flat_weights, self.bias, self.num_layers,
                             self.dropout, self.training, self.bidirectional, self.batch_first)
        else:
            result = _VF.gru(input, batch_sizes, hx, self._flat_weights, self.bias,
                             self.num_layers, self.dropout, self.training, self.bidirectional)
        output = result[0]
        hidden = result[1]

        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(
                output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            return output, self.permute_hidden(hidden, unsorted_indices)

    def _reset_parameters(self) -> None:
        for weight in self.parameters():
            weight.data.normal_(mean=0.0, std=self.config.init_variance)


class DynamicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False, rnn_type='LSTM'):
        """
        LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).

        :param input_size:The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layers:Number of recurrent layers.
        :param bias:If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first:If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout:If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional:If True, becomes a bidirectional RNN. Default: False
        :param rnn_type: {LSTM, GRU, RNN}
        """
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type

        if self.rnn_type == 'LSTM':
            self.RNN = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, x_len):
        """
        sequence -> sort -> pad and pack ->process using RNN -> unpack ->unsort

        :param x: sequence embedding vectors
        :param x_len: numpy/tensor list
        :return:
        """
        """sort"""
        x_sort_idx = torch.sort(-x_len)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(
            x, x_len, batch_first=self.batch_first)

        # process using the selected RNN
        if self.rnn_type == 'LSTM':
            out_pack, (ht, ct) = self.RNN(x_emb_p, None)
        else:
            out_pack, ht = self.RNN(x_emb_p, None)
            ct = None
        """unsort: h"""
        ht = torch.transpose(ht, 0, 1)[
            x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
        ht = torch.transpose(ht, 0, 1)

        if self.only_use_last_hidden_state:
            return ht
        else:
            """unpack: out"""
            out = torch.nn.utils.rnn.pad_packed_sequence(
                out_pack, batch_first=self.batch_first)  # (sequence, lengths)
            out = out[0]  #
            out = out[x_unsort_idx]
            """unsort: out c"""
            if self.rnn_type == 'LSTM':
                ct = torch.transpose(ct, 0, 1)[
                    x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
                ct = torch.transpose(ct, 0, 1)

            return out, (ht, ct)