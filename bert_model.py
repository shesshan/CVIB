from copy import deepcopy
import imp
from operator import add
from turtle import forward
from transformers import BertPreTrainedModel
from transformers.modeling_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    BaseModelOutput
)
import numpy as np
import torch.nn.functional as F
import logging
from dataclasses import dataclass
from typing import Optional, Tuple
from collections import OrderedDict
import math
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import init
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.activations import gelu, gelu_new
from model_utils import *
import logging

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "gelu_new": gelu_new}

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-large-uncased"
    # See all BERT models at https://huggingface.co/models?filter=bert
]


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(
            config.max_position_embeddings).expand((1, -1)))

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:,
                                             past_key_values_length: seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(
                input_ids)  # (bs,seq_len,hidden_size)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings  # (bs,seq_len,hidden_size)


class BertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * \
            self.attention_head_size  # [7.24 add] actually hidden size

        self.query = PrunableLinear(config.hidden_size, self.all_head_size)
        self.key = PrunableLinear(config.hidden_size, self.all_head_size)
        self.value = PrunableLinear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )  # default: position embeddings
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(
                self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(
                self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + \
                    relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1) # (bs , num_heads , seq_len , seq_len)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs) 

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer) # value_layer: (bs, seq_len, head_hidden_size)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # (bs, seq_len, num_heads, head_hidden_size)
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape) # (bs, seq_len, num_heads * head_hidden_size)

        outputs = (context_layer, attention_probs) if output_attentions else (
            context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutput(nn.Module):
    '''
    residual connections and layer normalization
    '''

    def __init__(self, config):
        super().__init__()
        self.dense = PrunableLinear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = BertSelfAttention(
            config, position_embedding_type=position_embedding_type)  # [7.24 add] output size: (bs,seq_len,hid_size)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - \
            len(heads)
        self.self.all_head_size = self.self.attention_head_size * \
            self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(  # (context_layer, attention_probs), context_layer: (bs, seq_len, hidden_size), attention_probs:(sen_len,1)
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        # add attentions if we output them
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = self.config.hidden_size
        intermediate_size = self.config.intermediate_size
        self.dense = PrunableLinear(hidden_size, intermediate_size)
        if isinstance(config.hidden_act, str):
            # default: gelu
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states, attention_mask=None, aspect_mask=None, aspect_embedding=None, sentence_embeddings=None):
        #assert aspect_mask is not None
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class Reweighting(nn.Module):
    def __init__(self, config, first_layer=False):
        super().__init__()
        self.config = config
        self.if_first_layer = first_layer
        hidden_size = self.config.hidden_size
        self.ws = nn.Parameter(
            torch.Tensor(hidden_size, hidden_size))
        if first_layer:
            self.whs = nn.Parameter(
                torch.Tensor(hidden_size, self.config.gru_hidden_size))
            self.wd = nn.Parameter(
                torch.Tensor(hidden_size, hidden_size))
        else:
            self.wd = nn.Parameter(
                torch.Tensor(self.config.gru_hidden_size, hidden_size))
        self.wa = nn.Parameter(
            torch.Tensor(hidden_size, hidden_size))
        self.activation = nn.Tanh()
        self.w = nn.Parameter(torch.Tensor(hidden_size))
        # init.uniform_(self.w, -1.0, 1.0)  # NOTE: [5.12 add] uniform w!
        self._init_weights()
        self.gru = DRGRU(config=self.config, input_size=hidden_size,
                         hidden_size=self.config.gru_hidden_size, batch_first=True, bias=False)

    def forward(self, sentence_representation, attention_mask=None, aspect_embedding=None, sentence_embeddings=None, layer_num=0):
        # logging.info('Reweighting {}'.format(attention_mask)) size，value都没问题了
        batch_size, seq_len, hidden_size = sentence_embeddings.size()
        # re-weighting function
        A = (sentence_representation.matmul(self.wd)+aspect_embedding.matmul(self.wa)
             ).unsqueeze(1).expand(batch_size, seq_len, hidden_size)
        S = sentence_embeddings.matmul(self.ws)
        M = S + A  # (bs,len,hid)
        # logging.info(M) 这里没问题
        #assert not np.any(np.array(np.isnan(M.cpu())))
        # logging.info(self.activation(M)) 这里没问题
        # logging.info(self.w)  w出现问题！包含nan
        #assert not np.any(np.array(np.isnan(self.w.cpu())))
        m = self.activation(M).matmul(self.w)  # weights, (bs,seq_len)
        m = F.softmax(self.config.lambda_m * m, dim=-1)  # after softmax
        #assert not np.any(np.isnan(m.detach().cpu()).numpy())
        # (bs,seq_len,hid)
        reweigted_embeddings = m.unsqueeze(-1) * sentence_embeddings
        assert not np.all(reweigted_embeddings.detach().cpu().numpy() == 0)
        #assert not np.any(np.isnan(reweigted_embeddings.detach().cpu()).numpy())
        # logging.info(attention_mask)
        if len(attention_mask.size()) > 2:
            for _ in range(len(attention_mask.size())-2):
                attention_mask = attention_mask.squeeze(
                    1)  # final size: (bs,hid)
        at = ((reweigted_embeddings * attention_mask.unsqueeze(-1)
               ).sum(1) / attention_mask.sum(-1).unsqueeze(-1))  # (bs,hid)
        # logging.info(at)
        #assert not np.any(np.isnan(at.detach().cpu()).numpy())
        # GRU
        if layer_num == 0:
            sentence_representation = sentence_representation.matmul(self.whs)
        _, sentence_representation = self.gru(
            at.unsqueeze(1), sentence_representation.unsqueeze(0))
        # assert not np.any(np.isnan(sentence_representation.detach().cpu()).numpy())
        sentence_representation = sentence_representation.view(
            -1, sentence_representation.size(-1))
        # logging.info(sentence_representation)
        return sentence_representation  # (bs,gru_hid)

    def _init_weights(self, init_variance=1e-3):
        if self.config.init_variance:
            init_variance = self.config.init_variance
        self.ws.data.normal_(mean=0.0, std=init_variance)  # S
        if self.if_first_layer:
            self.whs.data.normal_(
                mean=0.0, std=init_variance)  # initialized hs
        self.wd.data.normal_(mean=0.0, std=init_variance)  # hd
        self.wa.data.normal_(mean=0.0, std=init_variance)  # ha
        self.w.data.normal_(mean=0.0, std=init_variance)  # w


class DRA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # num of layers, default:7
        if self.config.dra_share_params:
            hidden_size = self.config.hidden_size
            gru_hidden_size = self.config.gru_hidden_size

            self.ws = nn.Parameter(
                torch.Tensor(hidden_size, hidden_size))
            self.wa = nn.Parameter(
                torch.Tensor(hidden_size, hidden_size))
            self.activation = nn.Tanh()
            self.w = nn.Parameter(torch.Tensor(hidden_size))
            # for the first layer
            self.whs = nn.Parameter(
                torch.Tensor(hidden_size, gru_hidden_size))
            self.wd1 = nn.Parameter(
                torch.Tensor(hidden_size, hidden_size))
            # for other 6 layers
            self.wd = nn.Parameter(
                torch.Tensor(gru_hidden_size, hidden_size))
            self.gru = DRGRU(config=self.config, input_size=hidden_size,
                             hidden_size=gru_hidden_size, batch_first=True, bias=False)
            self._init_weights()
        else:
            self.layers = nn.ModuleList(
                [Reweighting(config, first_layer=True)])
            for _ in range(int(config.num_gru_layers)-1):
                self.layers.append(Reweighting(config))

    def forward(self, sentence_representation, attention_mask=None, aspect_embedding=None, sentence_embeddings=None):
        '''
        sentence_representation: FFN output, i.e. initial hs (batch_size,hidden_size)
        aspect_embedding: aspect embedding, (batch_size,hidden_size)
        sentence_embeddings: sentence embeddings (batch_size,seq_len,hidden_size)

        return: 
        sentence_representation: DRA output, i.e. hT (batch_size,gru_hidden_size)
        '''
        if self.config.dra_share_params:
            return self.share_forward(sentence_representation=sentence_representation, attention_mask=attention_mask, aspect_embedding=aspect_embedding, sentence_embeddings=sentence_embeddings)
        else:
            return self.no_share_forward(sentence_representation=sentence_representation, attention_mask=attention_mask, aspect_embedding=aspect_embedding, sentence_embeddings=sentence_embeddings)

    def no_share_forward(self, sentence_representation, attention_mask=None, aspect_embedding=None, sentence_embeddings=None):
        for i, layer_module in enumerate(self.layers):
            sentence_representation = layer_module(
                sentence_representation, attention_mask=attention_mask, aspect_embedding=aspect_embedding, sentence_embeddings=sentence_embeddings, layer_num=i)
        return sentence_representation

    def share_forward(self, sentence_representation, attention_mask=None, aspect_embedding=None, sentence_embeddings=None):
        batch_size, seq_len, hidden_size = sentence_embeddings.size()
        for t in range(self.config.num_gru_layers):
            # re-weighting function
            if t == 0:
                Wd = self.wd1
            else:
                Wd = self.wd
            H = torch.stack(
                seq_len * [sentence_representation.matmul(Wd)], dim=1)
            A = torch.stack(
                seq_len * [aspect_embedding.matmul(self.wa)], dim=1)
            S = sentence_embeddings.matmul(self.ws)
            M = S + H + A  # (bs,len,hid)
            # weights, (bs,seq_len)
            m = self.activation(M).matmul(self.w)
            m = F.softmax(self.config.lambda_m * m, dim=-1)  # after softmax
            reweigted_embeddings = m.unsqueeze(-1) * sentence_embeddings
            assert not np.all(reweigted_embeddings.detach().cpu().numpy() == 0)
            if len(attention_mask.size()) > 2:
                for _ in range(len(attention_mask.size())-2):
                    attention_mask = attention_mask.squeeze(
                        1)  # final size: (bs,hid)
            at = ((reweigted_embeddings * attention_mask.unsqueeze(-1)
                   ).sum(1) / attention_mask.sum(-1).unsqueeze(-1))  # (bs,hid)
            # GRU
            if t == 0:
                sentence_representation = sentence_representation.matmul(
                    self.whs)
            _, sentence_representation = self.gru(
                at.unsqueeze(1), sentence_representation.unsqueeze(0))
            sentence_representation = sentence_representation.view(
                -1, sentence_representation.size(-1))
        return sentence_representation  # (bs,gru_hid)

    def _init_weights(self, init_variance=1e-6):
        if self.config.init_variance is not None:
            init_variance = self.config.init_variance
        self.ws.data.normal_(mean=0.0, std=init_variance)  # S
        self.whs.data.normal_(
            mean=0.0, std=init_variance)  # initialized hs
        self.wd1.data.normal_(
            mean=0.0, std=init_variance)
        self.wd.data.normal_(mean=0.0, std=init_variance)  # hd
        self.wa.data.normal_(mean=0.0, std=init_variance)  # ha
        self.w.data.normal_(mean=0.0, std=init_variance)  # w


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = self.config.hidden_size
        self.dense = PrunableLinear(
            self.config.intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(
            hidden_size, eps=self.config.layer_norm_eps)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, attention_mask=None, aspect_mask=None, aspect_embedding=None, sentence_embeddings=None):
        # feed-forward network output
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor  # residual connection
        hidden_states = self.LayerNorm(hidden_states)  # normalization
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config, layer_num=-1, add_vib=True):
        super().__init__()
        self.config = config
        hidden_size = self.config.hidden_size
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.original_attention_mask = None
        self.aspect_mask = None
        self.aspect_embedding = None
        self.sentence_embeddings = None
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(
                    f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(
                config, position_embedding_type="absolute")

        self.intermediate = BertIntermediate(config)  # feed-forward layer
        self.output = BertOutput(config)
        self.layer_num = layer_num
        self.add_vib = add_vib
        if self.config.use_ib and self.add_vib:
            logging.info(
                'Add VIB-based masking layer into the {}-th BERT layer'.format(int(self.layer_num)+1))
            self.ib = InformationBottleneck(hidden_size, mask_thresh=config.threshold, init_mag=config.init_mag, init_var=config.init_var,
                                            kl_mult=config.kl_mult, sample_in_training=config.sample_in_training, sample_in_testing=config.sample_in_testing)
        else:
            self.ib = None

    def get_vib_layer(self):
        return self.ib

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        original_attention_mask=None,
        aspect_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        aspect_embedding=None,
        sentence_embeddings=None,
    ):
        '''
        each layer adding:
            original_attention_mask: (bs,seq_len,hidden_size)
            aspect_embedding: original aspect averaged embedding (bs,hidden_size)
            sentence_embeddings: original sentence embeddings (bs,seq_len,hidden_size)
        '''
        # [5.12 add]
        self.original_attention_mask = original_attention_mask
        # logging.info('BertLayer {}'.format(original_attention_mask))
        # [5.10 add]
        #assert aspect_mask is not None
        if aspect_mask is not None:
            self.aspect_mask = aspect_mask
        # [5.11 add]
        if aspect_embedding is not None:
            self.aspect_embedding = aspect_embedding
        if sentence_embeddings is not None:
            self.sentence_embeddings = sentence_embeddings

        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:
                                                  2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            # add self attentions if we output attention weights
            outputs = self_attention_outputs[1:]

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:
                                                       ] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            # add cross attentions if we output attention weights
            outputs = outputs + cross_attention_outputs[1:-1]

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )  # NOTE: feed-forward layer
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)
        return outputs

    def feed_forward_chunk(self, attention_output):
        '''
        feed forward network
        '''
        intermediate_output = self.intermediate(
            attention_output, attention_mask=self.original_attention_mask,
            aspect_mask=self.aspect_mask, aspect_embedding=self.aspect_embedding, sentence_embeddings=self.sentence_embeddings)
        layer_output = self.output(intermediate_output, attention_output, attention_mask=self.original_attention_mask,
                                   aspect_mask=self.aspect_mask, aspect_embedding=self.aspect_embedding, sentence_embeddings=self.sentence_embeddings)
        # [7.23 add] VIB layer
        if self.config.use_ib and self.add_vib:
            layer_output = self.ib(layer_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer, self.kl_list = self.make_bert_layers(self.config)
        self.gradient_checkpointing = False

    def get_kl_list(self):
        return self.kl_list

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        original_attention_mask=None,
        aspect_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        '''
        hidden_states: embdedings from Embedding Layer (bs,seq_len,hidden_size)
        original_attention_mask: (bs,seq_len)
        aspect_mask: (bs,seq_len)
        '''
        # 从这里开始attention mask就变为extended attention mask了，主要用于self-attention计算
        # 因此需要引入original attention mask，用来对FFN的output做Max pooling
        # logging.info('bert encoder: {}'.format(original_attention_mask))
        # [5.11 add] abstract aspect averaged embedding
        '''
        if aspect_mask is not None:
            sentence_embeddings = hidden_states
            expand_aspect_mask = (
                1 - aspect_mask).unsqueeze(-1).bool()  # (bs,seq_len,1)
            # aspect averaged  embedding (bs,hidden_size)
            aspect_embedding = torch.div(torch.sum(hidden_states.masked_fill(expand_aspect_mask, 0), dim=-2),
                                         torch.sum(aspect_mask.float(), dim=-1).unsqueeze(-1))
        '''
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None

        for i, layer_module in enumerate(self.layer):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logging.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    aspect_mask=aspect_mask,
                    original_attention_mask=original_attention_mask,
                    head_mask=layer_head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    aspect_mask=aspect_mask,
                    original_attention_mask=original_attention_mask,
                    head_mask=layer_head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + \
                        (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

    def make_bert_layers(self, config):
        layers, kl_list = [], []
        add_vib = True
        for i in range(self.config.num_hidden_layers):
            # [9.4 add] only first/last 3 VIB layers
            if (self.config.first3 and i >= 3) or (self.config.last3 and i <= 8):
                add_vib = False
            else:
                add_vib = True
            bert_layer = BertLayer(config, layer_num=i, add_vib=add_vib)
            layers.append(bert_layer)
            if self.config.use_ib and add_vib:
                ib = bert_layer.get_vib_layer()
                kl_list.append(ib)
        return nn.ModuleList(layers), kl_list


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = PrunableLinear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]  # NOTE: [cls] size:(N,D)
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.kl_list = None
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def set_ib_flag(self, use_ib=True):
        self.config.use_ib = use_ib

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        aspect_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        input_ids: (bs,seq_len)
        attention_mask: (bs,seq_len)
        aspect_mask: (bs,seq_len)
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        # logging.info('bert model: {}'.format(attention_mask)) 这里是原始的attention mask，没问题
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device)
        # logging.info(extended_attention_mask)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            original_attention_mask=attention_mask,
            aspect_mask=aspect_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]  # (N,L,D)
        pooled_output = self.pooler(
            sequence_output) if self.pooler is not None else None

        outputs = (sequence_output, pooled_output, self.encoder.get_kl_list())
        if self.config.output_attentions:
            outputs = outputs + (encoder_outputs['attentions'],)

        return outputs


@dataclass
class VIBBERTOutput(BaseModelOutput):
    last_hidden_state: torch.FloatTensor = None


@dataclass
class ABSAOutput(BaseModelOutput):
    ce_loss: torch.FloatTensor = None
    pooler_output_1: torch.FloatTensor = None
    pooler_output_2: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
