import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.nn.init import normal_
from einops import rearrange, repeat
from functools import partial

from recbole.utils import FeatureType, FeatureSource
from recbole.model.layers import FeedForward, MultiHeadAttention


### MLPMixer ###
class MLPMixerEncoder(nn.Module):
    def __init__(self,
                 n_layers=2,
                 hidden_size=64,
                 seq_len=50,
                 inner_size=256,
                 hidden_dropout_prob=0.5,
                 hidden_act='gelu',
                 layer_norm_eps=1e-12):
        super(MLPMixerEncoder, self).__init__()

        layer = MLPMixerLayer(hidden_size, seq_len, inner_size,
                              hidden_dropout_prob, hidden_act, layer_norm_eps)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(n_layers)])

    def forward(self, hidden_states, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class MLPMixerLayer(nn.Module):
    def __init__(self, hidden_size, seq_len, intermediate_size,
                 hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(MLPMixerLayer, self).__init__()
        self.chan_first, self.chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
        self.expansion_factor = intermediate_size // hidden_size
        self.pre1 = PreNormResidual(hidden_size, FlexFeedForward(seq_len, self.expansion_factor,
                                                                 hidden_dropout_prob, hidden_act,
                                                                 layer_norm_eps, self.chan_first), layer_norm_eps)
        self.pre2 = PreNormResidual(hidden_size, FlexFeedForward(hidden_size, self.expansion_factor,
                                                                 hidden_dropout_prob, hidden_act,
                                                                 layer_norm_eps, self.chan_last), layer_norm_eps)

    def forward(self, hidden_states):
        item_mixer_output = self.pre1(hidden_states)
        channel_mixer_output = self.pre2(item_mixer_output)
        return channel_mixer_output


class PreNormResidual(nn.Module):
    def __init__(self, hidden_size, ffn, layer_norm_eps):
        super(PreNormResidual, self).__init__()
        self.ffn = ffn
        self.ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states):
        return self.ffn(self.ln(hidden_states)) + hidden_states


class FlexFeedForward(nn.Module):
    def __init__(self, dim, expansion_factor, dropout_prob, hidden_act, layer_norm_eps, dense=nn.Linear):
        super(FlexFeedForward, self).__init__()
        self.dense_1 = dense(dim, dim * expansion_factor)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = dense(dim * expansion_factor, dim)
        self.LayerNorm = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": fn.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


### gMLP ###
class gMLPEncoder(nn.Module):
    def __init__(self,
                 n_layers=2,
                 hidden_size=64,
                 seq_len=50,
                 inner_size=256,
                 hidden_dropout_prob=0.5,
                 hidden_act='gelu',
                 layer_norm_eps=1e-12):
        super(gMLPEncoder, self).__init__()
        layer = gMLPLayer(hidden_size, seq_len, inner_size,
                          hidden_dropout_prob, hidden_act, layer_norm_eps)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(n_layers)])

    def forward(self, hidden_states, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class gMLPLayer(nn.Module):
    def __init__(self, hidden_size, seq_len, intermediate_size,
                 hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(gMLPLayer, self).__init__()
        self.ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.proj_1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU()
        self.spatial_gating_unit = SpatialGatingUnit(intermediate_size, seq_len)
        self.proj_2 = nn.Linear(intermediate_size // 2, hidden_size)

    def forward(self, hidden_states):
        shorcut = hidden_states.clone()
        hidden_states = self.ln(hidden_states)
        hidden_states = self.proj_1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.spatial_gating_unit(hidden_states)
        hidden_states = self.proj_2(hidden_states)
        return hidden_states + shorcut


class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super(SpatialGatingUnit, self).__init__()
        self.ln = nn.LayerNorm(d_ffn // 2)
        self.proj = nn.Linear(seq_len, seq_len)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.ln(v)
        v = v.permute(0, 2, 1)
        v = self.proj(v)
        v = v.permute(0, 2, 1)
        return u * v


### Linformer ###
class LinearMap(nn.Module):
    def __init__(self, seq_len, k_heads):
        super().__init__()
        self.k_heads = k_heads  # k head
        # self.theta_k = nn.Parameter(torch.randn([self.k_heads, seq_len]))
        self.lin = nn.Linear(seq_len, self.k_heads, bias=True)
        torch.nn.init.xavier_normal_(self.lin.weight)

    def forward(self, input_tensor):  # [B, L, d] -> [B, k, d]
        # result = torch.matmul(self.theta_k, input_tensor)
        result = self.lin(input_tensor.transpose(1, 2))
        return result.transpose(1, 2)


class LinformerAttention(nn.Module):

    def __init__(self, config, n_heads, seq_len, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super(LinformerAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads))
        k_heads = config['k_heads']
        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)  # 64
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.linear_key = LinearMap(seq_len, k_heads)
        self.linear_value = LinearMap(seq_len, k_heads)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask=None):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(self.linear_key(mixed_key_layer))
        value_layer = self.transpose_for_scores(self.linear_value(mixed_value_layer))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [1024, 2, 50, 50]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores  # + attention_mask #+ abs_pos_bias

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)  # [1024, 2, 50, 32]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [1024, 50, 2, 32]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # [1024, 50, 64]
        context_layer = context_layer.view(*new_context_layer_shape)  # [1024, 50, 64]

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


### Common Light Layer and Encoder ###
class LightAttentionTransformerLayer(nn.Module):

    def __init__(self, config, n_heads, seq_len, hidden_size, intermediate_size,
                 hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps):
        super(LightAttentionTransformerLayer, self).__init__()
        model_name = config['model']
        if model_name == 'Linformer':
            self.multi_head_attention = LinformerAttention(config, n_heads, seq_len, hidden_size,
                                                           hidden_dropout_prob, attn_dropout_prob, layer_norm_eps)
        elif model_name == 'Performer':
            self.multi_head_attention = PerformerAttention(config, n_heads, seq_len, hidden_size,
                                                           hidden_dropout_prob, attn_dropout_prob, layer_norm_eps)
        elif model_name == 'Synthesizer':
            self.multi_head_attention = SynthesizerAttention(n_heads, seq_len, hidden_size,
                                                             hidden_dropout_prob, attn_dropout_prob, layer_norm_eps)
        elif model_name == 'LinearTrm':
            self.multi_head_attention = LinearTrmAttention(n_heads, seq_len, hidden_size,
                                                           hidden_dropout_prob, attn_dropout_prob, layer_norm_eps)
        else:
            self.multi_head_attention = MultiHeadAttention(n_heads, seq_len, hidden_size,
                                                           hidden_dropout_prob, attn_dropout_prob, layer_norm_eps)
        self.feed_forward = FeedForward(hidden_size, intermediate_size,
                                        hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class LightAttentionTransformerEncoder(nn.Module):

    def __init__(self,
                 config,
                 n_layers=2,
                 n_heads=2,
                 seq_len=50,
                 hidden_size=64,
                 inner_size=256,
                 hidden_dropout_prob=0.5,
                 attn_dropout_prob=0.5,
                 hidden_act='gelu',
                 layer_norm_eps=1e-12):

        super(LightAttentionTransformerEncoder, self).__init__()

        layer = LightAttentionTransformerLayer(config, n_heads, seq_len, hidden_size, inner_size,
                                               hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask=None, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


## TiSASRecR
class TimeAwareRMultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,
                 use_order, use_distance):
        super(TimeAwareRMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.use_order = use_order
        self.use_distance = use_distance

        if use_order:
            self.order_affine = nn.Linear(2 * self.attention_head_size,1)
            self.activation = nn.Sigmoid()
        if use_distance:
            self.distance_affine = nn.Linear(2 * self.attention_head_size,1)
            self.scalar= nn.Parameter(torch.randn(1))

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask, absolute_pos_K, absolute_pos_V, time_matrix_emb_K,
                time_matrix_emb_V):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        absolute_pos_K_ = self.transpose_for_scores(absolute_pos_K).permute(0, 2, 3, 1)
        absolute_pos_V_ = self.transpose_for_scores(absolute_pos_V).permute(0, 2, 1, 3)
        time_matrix_emb_K_ = self.transpose_for_scores(time_matrix_emb_K).permute(0, 3, 1, 2,
                                                                                  4)  # [B, n_heads, L, L, D/n_heads]
        time_matrix_emb_V_ = self.transpose_for_scores(time_matrix_emb_V).permute(0, 3, 1, 2,
                                                                                  4)  # [B, n_heads, L, L, D/n_heads]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores_pos = torch.matmul(query_layer, absolute_pos_K_)
        attention_scores_time = torch.matmul(time_matrix_emb_K_, query_layer.unsqueeze(-1)).squeeze(
            -1)  # [B, n_heads, L, L, 1] -> [B, n_heads, L, L]

        max_seq_len = input_tensor.shape[-2]
        batch_size = input_tensor.shape[0]

        # generate concatenation
        key_layer_ = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 1, 3)
        q_vec = query_layer.unsqueeze(3).expand(batch_size, self.num_attention_heads, max_seq_len, max_seq_len, -1)
        k_vec = key_layer_.unsqueeze(2).expand(batch_size, self.num_attention_heads, max_seq_len, max_seq_len, -1)
        q_k = torch.cat((q_vec, k_vec), dim=-1)

        error_order = torch.zeros(attention_scores.shape).to(attention_scores.device)
        error_distance = torch.zeros(attention_scores.shape).to(attention_scores.device)

        if self.use_order:
            # Generate order ground truth
            gd_order = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).unsqueeze(0).unsqueeze(0)
            gd_order = gd_order.expand(batch_size, self.num_attention_heads, max_seq_len, max_seq_len).to(
                input_tensor.device)
            pr_order = self.activation(self.order_affine(q_k).squeeze(-1))
            error_order = torch.log(pr_order + 1e-24) * gd_order + torch.log(1 - pr_order + 1e-24) * (1 - gd_order)
        if self.use_distance:
            row_index = torch.arange(0, max_seq_len, 1).unsqueeze(0).expand((max_seq_len, max_seq_len))
            col_index = torch.arange(0, max_seq_len, 1).unsqueeze(1).expand((max_seq_len, max_seq_len))
            gd_distance = torch.log(torch.abs(row_index - col_index) + 1).unsqueeze(0).unsqueeze(0)
            gd_distance = gd_distance.expand(batch_size, self.num_attention_heads, max_seq_len, max_seq_len).to(
                input_tensor.device)
            pr_distance = self.distance_affine(q_k).squeeze(-1)
            error_distance = -torch.square(gd_distance - pr_distance) * torch.square(self.scalar) / 2

        attention_scores = attention_scores + attention_scores_pos + attention_scores_time \
                           + error_order + error_distance

        attention_scores = attention_scores / self.sqrt_attention_head_size
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer_pos = torch.matmul(attention_probs, absolute_pos_V_)
        context_layer_time = torch.matmul(attention_probs.unsqueeze(-2), time_matrix_emb_V_).squeeze(-2)
        context_layer = context_layer + context_layer_pos + context_layer_time

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TimeAwareRTransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
            self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
            layer_norm_eps, use_order, use_distance
    ):
        super(TimeAwareRTransformerLayer, self).__init__()
        self.multi_head_attention = TimeAwareRMultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,
            use_order=use_order,
            use_distance=use_distance
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states, attention_mask, absolute_pos_K, absolute_pos_V, time_matrix_emb_K,
                time_matrix_emb_V):
        attention_output = self.multi_head_attention(hidden_states, attention_mask, absolute_pos_K, absolute_pos_V,
                                                     time_matrix_emb_K, time_matrix_emb_V)
        feedforward_output = self.feed_forward(attention_output)

        return feedforward_output


class TimeAwareRTransformerEncoder(nn.Module):
    r""" One TransformerEncoder consists of several TransformerLayers.

    Args:
        n_layers(num): num of transformer layers in transformer encoder. Default: 2
        n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        hidden_size(num): the input and output hidden size. Default: 64
        inner_size(num): the dimensionality in feed-forward layer. Default: 256
        hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(
            self,
            n_layers=2,
            n_heads=2,
            hidden_size=64,
            inner_size=256,
            hidden_dropout_prob=0.5,
            attn_dropout_prob=0.5,
            hidden_act='gelu',
            layer_norm_eps=1e-12,
            use_order=True,
            use_distance=True
    ):

        super(TimeAwareRTransformerEncoder, self).__init__()
        layer = TimeAwareRTransformerLayer(
            n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps,
            use_order=use_order, use_distance=use_distance
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states,
                attention_mask,
                absolute_pos_K,
                absolute_pos_V,
                time_matrix_emb_K,
                time_matrix_emb_V,
                output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask, absolute_pos_K,
                                         absolute_pos_V, time_matrix_emb_K, time_matrix_emb_V)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


## TiSASRec
class TimeAwareMultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super(TimeAwareMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask, absolute_pos_K, absolute_pos_V, time_matrix_emb_K,
                time_matrix_emb_V):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        absolute_pos_K_ = self.transpose_for_scores(absolute_pos_K).permute(0, 2, 3, 1)
        absolute_pos_V_ = self.transpose_for_scores(absolute_pos_V).permute(0, 2, 1, 3)
        time_matrix_emb_K_ = self.transpose_for_scores(time_matrix_emb_K).permute(0, 3, 1, 2,
                                                                                  4)  # [B, n_heads, L, L, D/n_heads]
        time_matrix_emb_V_ = self.transpose_for_scores(time_matrix_emb_V).permute(0, 3, 1, 2,
                                                                                  4)  # [B, n_heads, L, L, D/n_heads]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores_pos = torch.matmul(query_layer, absolute_pos_K_)
        attention_scores_time = torch.matmul(time_matrix_emb_K_, query_layer.unsqueeze(-1)).squeeze(
            -1)  # [B, n_heads, L, L, 1] -> [B, n_heads, L, L]

        attention_scores = attention_scores + attention_scores_pos + attention_scores_time

        attention_scores = attention_scores / self.sqrt_attention_head_size
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer_pos = torch.matmul(attention_probs, absolute_pos_V_)
        context_layer_time = torch.matmul(attention_probs.unsqueeze(-2), time_matrix_emb_V_).squeeze(-2)
        context_layer = context_layer + context_layer_pos + context_layer_time

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TimeAwareTransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
            self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
            layer_norm_eps
    ):
        super(TimeAwareTransformerLayer, self).__init__()
        self.multi_head_attention = TimeAwareMultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states, attention_mask, absolute_pos_K, absolute_pos_V, time_matrix_emb_K,
                time_matrix_emb_V):
        attention_output = self.multi_head_attention(hidden_states, attention_mask, absolute_pos_K, absolute_pos_V,
                                                     time_matrix_emb_K, time_matrix_emb_V)
        feedforward_output = self.feed_forward(attention_output)

        return feedforward_output


class TimeAwareTransformerEncoder(nn.Module):
    r""" One TransformerEncoder consists of several TransformerLayers.

    Args:
        n_layers(num): num of transformer layers in transformer encoder. Default: 2
        n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        hidden_size(num): the input and output hidden size. Default: 64
        inner_size(num): the dimensionality in feed-forward layer. Default: 256
        hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(
            self,
            n_layers=2,
            n_heads=2,
            hidden_size=64,
            inner_size=256,
            hidden_dropout_prob=0.5,
            attn_dropout_prob=0.5,
            hidden_act='gelu',
            layer_norm_eps=1e-12
    ):

        super(TimeAwareTransformerEncoder, self).__init__()
        layer = TimeAwareTransformerLayer(
            n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states,
                attention_mask,
                absolute_pos_K,
                absolute_pos_V,
                time_matrix_emb_K,
                time_matrix_emb_V,
                output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask, absolute_pos_K,
                                         absolute_pos_V, time_matrix_emb_K, time_matrix_emb_V)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


# AC-SR
class AttackRMultiHeadAttention(nn.Module):
    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,
                 use_order, use_distance):
        super(AttackRMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.use_order = use_order
        self.use_distance = use_distance
        if self.use_order:
            self.order_affine = nn.Linear(2 * self.attention_head_size, 1)
            self.activation = nn.Sigmoid()
        if self.use_distance:
            self.distance_affine = nn.Linear(2 * self.attention_head_size, 1)
            self.scalar = nn.Parameter(torch.randn(1))

        self.attack_query_transform = nn.Linear(self.all_head_size, self.all_head_size)
        self.attack_key_transform = nn.Linear(self.all_head_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def cal_attack_mask(self, query, key, attention_mask):
        mixed_query_layer = self.attack_query_transform(query)
        mixed_key_layer = self.attack_key_transform(key)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)

        attention_scores = torch.matmul(query_layer, key_layer)

        attention_scores = attention_scores / self.sqrt_attention_head_size

        attention_scores = attention_scores + attention_mask

        attention_probs = self.softmax(attention_scores)

        attention_probs = self.attn_dropout(attention_probs)

        return attention_probs

    def cal_adjusted_outputs(self, attention_probs, input_tensor, value_layer):
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    def cal_origin_qkv(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        attention_scores = torch.matmul(query_layer, key_layer)

        origin_attention_scores = attention_scores
        max_seq_len = input_tensor.shape[-2]
        batch_size = input_tensor.shape[0]
        '''
            yqc modify here 0725
            add rich attention
        '''
        # generate concatenation
        key_layer_ = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 1, 3)
        q_vec = query_layer.unsqueeze(3).expand(batch_size, self.num_attention_heads, max_seq_len, max_seq_len, -1)
        k_vec = key_layer_.unsqueeze(2).expand(batch_size, self.num_attention_heads, max_seq_len, max_seq_len, -1)
        q_k = torch.cat((q_vec, k_vec), dim=-1)

        error_order = torch.zeros(attention_scores.shape).to(attention_scores.device)
        error_distance = torch.zeros(attention_scores.shape).to(attention_scores.device)

        if self.use_order:
            # Generate order ground truth
            gd_order = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).unsqueeze(0).unsqueeze(0)
            gd_order = gd_order.expand(batch_size, self.num_attention_heads, max_seq_len, max_seq_len).to(
                input_tensor.device)
            pr_order = self.activation(self.order_affine(q_k).squeeze(-1))
            error_order = torch.log(pr_order + 1e-24) * gd_order + torch.log(1 - pr_order + 1e-24) * (1 - gd_order)
        if self.use_distance:
            row_index = torch.arange(0, max_seq_len, 1).unsqueeze(0).expand((max_seq_len, max_seq_len))
            col_index = torch.arange(0, max_seq_len, 1).unsqueeze(1).expand((max_seq_len, max_seq_len))
            gd_distance = torch.log(torch.abs(row_index - col_index) + 1).unsqueeze(0).unsqueeze(0)
            gd_distance = gd_distance.expand(batch_size, self.num_attention_heads, max_seq_len, max_seq_len).to(
                input_tensor.device)
            pr_distance = self.distance_affine(q_k).squeeze(-1)
            error_distance = -torch.square(gd_distance - pr_distance) * torch.square(self.scalar) / 2

        attention_scores = attention_scores + error_order + error_distance

        def _func(_scores):
            _scores = _scores / self.sqrt_attention_head_size

            _scores = _scores + attention_mask
            _prob = self.softmax(_scores)
            _prob = self.attn_dropout(_prob)
            return _prob

        attention_probs = _func(attention_scores)
        origin_attention_probs = _func(origin_attention_scores)

        return mixed_query_layer, mixed_key_layer, value_layer, attention_probs, origin_attention_probs


class AttackRTransformerLayer(nn.Module):
    def __init__(
            self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
            layer_norm_eps, combine_option='fixed', use_order=True, use_distance=True, two_level=True,
            rich_calibrated_combine='fixed'
    ):
        super(AttackRTransformerLayer, self).__init__()
        self.hidden_size = hidden_size

        self.attack_attention = AttackRMultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,
            use_order=use_order, use_distance=use_distance
        )
        self.two_level = two_level
        self.rich_calibrated_combine = rich_calibrated_combine
        if self.rich_calibrated_combine == 'trainable':
            self.rich_calibrated_combine_ratio = torch.nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.combine_option = combine_option
        if self.combine_option == 'gate':
            self.gate = torch.nn.Linear(hidden_size, 50)
        self.combine_ratio = 0.5
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)
        self.anneal_step = 0

    def combine_attention(self, origin_attention, calibrated_attention, query):
        if self.combine_option == 'fixed':
            combined_attention = torch.softmax(origin_attention + self.combine_ratio * calibrated_attention, dim=-1)
        elif self.combine_option == 'gate':
            combine_gate = torch.sigmoid(self.gate(query)).unsqueeze(1)
            combined_attention = combine_gate * origin_attention + (1 - combine_gate) * calibrated_attention
        elif self.combine_option == 'annealing':
            adjust_rate = np.exp(-self.anneal_step / 100000)
            self.anneal_step += 1
            combined_attention = adjust_rate * origin_attention + (1 - adjust_rate) * calibrated_attention

        else:
            raise KeyError
        return combined_attention

    def forward(self, hidden_states, attention_mask, return_attention_prob=False):

        mixed_query, mixed_key, value, after_rich_attn_probs, before_rich_attention_probs = \
            self.attack_attention.cal_origin_qkv(hidden_states, attention_mask)
        if self.two_level:
            origin_attn_probs = after_rich_attn_probs
        else:
            origin_attn_probs = before_rich_attention_probs
        attack_mask = self.attack_attention.cal_attack_mask(mixed_query, mixed_key, attention_mask)
        noise = torch.randn(attack_mask.shape).to(attack_mask.device)
        attacked_attention_prob = origin_attn_probs * attack_mask + noise * (1 - attack_mask)

        calibrated_attention_prob = origin_attn_probs * torch.exp(1 - attack_mask)

        combined_attention_prob = self.combine_attention(origin_attn_probs,
                                                         calibrated_attention_prob,
                                                         mixed_query)

        if not self.two_level:
            if self.rich_calibrated_combine == 'fixed':
                combined_attention_prob = (combined_attention_prob + after_rich_attn_probs) / 2
            elif self.rich_calibrated_combine == 'trainable':
                combined_attention_prob = self.rich_calibrated_combine_ratio * combined_attention_prob + \
                                          (1 - self.rich_calibrated_combine_ratio) * after_rich_attn_probs
            else:
                raise KeyError

        attacked_attention_output = self.attack_attention.cal_adjusted_outputs(attacked_attention_prob,
                                                                               hidden_states,
                                                                               value)

        calibrated_attention_output = self.attack_attention.cal_adjusted_outputs(combined_attention_prob,
                                                                                 hidden_states,
                                                                                 value)

        attacked_feedforward_output = self.feed_forward(attacked_attention_output)
        calibrated_feedforward_output = self.feed_forward(calibrated_attention_output)
        if return_attention_prob:
            return attacked_feedforward_output, calibrated_feedforward_output, attack_mask, combined_attention_prob

        return attacked_feedforward_output, calibrated_feedforward_output, attack_mask


class AttackRTransformerEncoder(nn.Module):
    def __init__(
            self,
            n_layers=2,
            n_heads=2,
            hidden_size=64,
            inner_size=256,
            hidden_dropout_prob=0.5,
            attn_dropout_prob=0.5,
            hidden_act='gelu',
            layer_norm_eps=1e-12,
            combine_option='fixed',
            use_order=True,
            use_distance=True,
            two_level=True,
            rich_calibrated_combine='fixed'
    ):

        super(AttackRTransformerEncoder, self).__init__()
        layer = AttackRTransformerLayer(
            n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps,
            combine_option, use_order=use_order, use_distance=use_distance, two_level=two_level,
            rich_calibrated_combine=rich_calibrated_combine
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask,
                output_all_encoded_layers=True, return_attention_prob=False):
        all_encoder_layers = []
        attacked_hidden_states = None
        calibrated_hidden_states = None
        all_attack_masks = []
        all_attention_prob = [] if return_attention_prob else None
        for layer_module in self.layer:
            if return_attention_prob:
                attacked_hidden_states, calibrated_hidden_states, attack_mask, combined_attention_prob = \
                    layer_module(hidden_states, attention_mask, return_attention_prob)
            else:
                attacked_hidden_states, calibrated_hidden_states, attack_mask = \
                    layer_module(hidden_states, attention_mask, return_attention_prob)
            # print(attack_mask.shape)
            hidden_states = calibrated_hidden_states
            all_attack_masks.append(attack_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append((attacked_hidden_states, calibrated_hidden_states))
            if return_attention_prob:
                all_attention_prob.append(combined_attention_prob)
        if not output_all_encoded_layers:
            all_encoder_layers.append((attacked_hidden_states, calibrated_hidden_states))
        if return_attention_prob:
            return all_encoder_layers, all_attack_masks, all_attention_prob
        return all_encoder_layers, all_attack_masks


# ACTiSASRec
class ACTimeAwareMultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,
                 use_order, use_distance):
        super(ACTimeAwareMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.use_order = use_order
        self.use_distance = use_distance
        if self.use_order:
            self.order_affine = nn.Linear(2 * self.attention_head_size, 1)
            self.activation = nn.Sigmoid()
        if self.use_distance:
            self.distance_affine = nn.Linear(2 * self.attention_head_size, 1)
            self.scalar = nn.Parameter(torch.randn(1))

        self.attack_query_transform = nn.Linear(self.all_head_size, self.all_head_size)
        self.attack_key_transform = nn.Linear(self.all_head_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def cal_attack_mask(self, query, key, attention_mask):
        mixed_query_layer = self.attack_query_transform(query)
        mixed_key_layer = self.attack_key_transform(key)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)

        attention_scores = torch.matmul(query_layer, key_layer)

        attention_scores = attention_scores / self.sqrt_attention_head_size

        attention_scores = attention_scores + attention_mask

        attention_probs = self.softmax(attention_scores)

        attention_probs = self.attn_dropout(attention_probs)

        return attention_probs

    def cal_adjusted_outputs(self, attention_probs, input_tensor, value_layer):
        (value_layer, absolute_pos_V_, time_matrix_emb_V_) = value_layer

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer_pos = torch.matmul(attention_probs, absolute_pos_V_)
        context_layer_time = torch.matmul(attention_probs.unsqueeze(-2), time_matrix_emb_V_).squeeze(-2)
        context_layer = context_layer + context_layer_pos + context_layer_time

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # context_layer = torch.matmul(attention_probs, value_layer)
        # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # context_layer = context_layer.view(*new_context_layer_shape)
        # hidden_states = self.dense(context_layer)
        # hidden_states = self.out_dropout(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    def cal_origin_qkv(self, input_tensor, attention_mask,
                       absolute_pos_K, absolute_pos_V, time_matrix_emb_K, time_matrix_emb_V):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        absolute_pos_K_ = self.transpose_for_scores(absolute_pos_K).permute(0, 2, 3, 1)
        absolute_pos_V_ = self.transpose_for_scores(absolute_pos_V).permute(0, 2, 1, 3)
        time_matrix_emb_K_ = self.transpose_for_scores(time_matrix_emb_K).permute(0, 3, 1, 2,
                                                                                  4)  # [B, n_heads, L, L, D/n_heads]
        time_matrix_emb_V_ = self.transpose_for_scores(time_matrix_emb_V).permute(0, 3, 1, 2,
                                                                                  4)  # [B, n_heads, L, L, D/n_heads]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores_pos = torch.matmul(query_layer, absolute_pos_K_)
        attention_scores_time = torch.matmul(time_matrix_emb_K_, query_layer.unsqueeze(-1)).squeeze(
            -1)  # [B, n_heads, L, L, 1] -> [B, n_heads, L, L]

        attention_scores = attention_scores + attention_scores_pos + attention_scores_time

        # attention_scores = torch.matmul(query_layer, key_layer)

        origin_attention_scores = attention_scores
        max_seq_len = input_tensor.shape[-2]
        batch_size = input_tensor.shape[0]
        '''
            yqc modify here 0725
            add rich attention
        '''
        # generate concatenation
        key_layer_ = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 1, 3)
        q_vec = query_layer.unsqueeze(3).expand(batch_size, self.num_attention_heads, max_seq_len, max_seq_len, -1)
        k_vec = key_layer_.unsqueeze(2).expand(batch_size, self.num_attention_heads, max_seq_len, max_seq_len, -1)
        q_k = torch.cat((q_vec, k_vec), dim=-1)

        error_order = torch.zeros(attention_scores.shape).to(attention_scores.device)
        error_distance = torch.zeros(attention_scores.shape).to(attention_scores.device)

        if self.use_order:
            # Generate order ground truth
            gd_order = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).unsqueeze(0).unsqueeze(0)
            gd_order = gd_order.expand(batch_size, self.num_attention_heads, max_seq_len, max_seq_len).to(
                input_tensor.device)
            pr_order = self.activation(self.order_affine(q_k).squeeze(-1))
            error_order = torch.log(pr_order + 1e-24) * gd_order + torch.log(1 - pr_order + 1e-24) * (1 - gd_order)
        if self.use_distance:
            row_index = torch.arange(0, max_seq_len, 1).unsqueeze(0).expand((max_seq_len, max_seq_len))
            col_index = torch.arange(0, max_seq_len, 1).unsqueeze(1).expand((max_seq_len, max_seq_len))
            gd_distance = torch.log(torch.abs(row_index - col_index) + 1).unsqueeze(0).unsqueeze(0)
            gd_distance = gd_distance.expand(batch_size, self.num_attention_heads, max_seq_len, max_seq_len).to(
                input_tensor.device)
            pr_distance = self.distance_affine(q_k).squeeze(-1)
            error_distance = -torch.square(gd_distance - pr_distance) * torch.square(self.scalar) / 2

        attention_scores = attention_scores + error_order + error_distance

        def _func(_scores):
            _scores = _scores / self.sqrt_attention_head_size

            _scores = _scores + attention_mask
            _prob = self.softmax(_scores)
            _prob = self.attn_dropout(_prob)
            return _prob

        attention_probs = _func(attention_scores)
        origin_attention_probs = _func(origin_attention_scores)

        return mixed_query_layer, mixed_key_layer, (value_layer, absolute_pos_V_, time_matrix_emb_V_), \
               attention_probs, origin_attention_probs

    # def forward(self, input_tensor, attention_mask, absolute_pos_K, absolute_pos_V, time_matrix_emb_K,
    #             time_matrix_emb_V):
    #     mixed_query_layer = self.query(input_tensor)
    #     mixed_key_layer = self.key(input_tensor)
    #     mixed_value_layer = self.value(input_tensor)
    #
    #     query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
    #     key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
    #     value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)
    #
    #     absolute_pos_K_ = self.transpose_for_scores(absolute_pos_K).permute(0, 2, 3, 1)
    #     absolute_pos_V_ = self.transpose_for_scores(absolute_pos_V).permute(0, 2, 1, 3)
    #     time_matrix_emb_K_ = self.transpose_for_scores(time_matrix_emb_K).permute(0, 3, 1, 2,
    #                                                                               4)  # [B, n_heads, L, L, D/n_heads]
    #     time_matrix_emb_V_ = self.transpose_for_scores(time_matrix_emb_V).permute(0, 3, 1, 2,
    #                                                                               4)  # [B, n_heads, L, L, D/n_heads]
    #
    #     # Take the dot product between "query" and "key" to get the raw attention scores.
    #     attention_scores = torch.matmul(query_layer, key_layer)
    #     attention_scores_pos = torch.matmul(query_layer, absolute_pos_K_)
    #     attention_scores_time = torch.matmul(time_matrix_emb_K_, query_layer.unsqueeze(-1)).squeeze(
    #         -1)  # [B, n_heads, L, L, 1] -> [B, n_heads, L, L]
    #
    #     attention_scores = attention_scores + attention_scores_pos + attention_scores_time
    #
    #     attention_scores = attention_scores / self.sqrt_attention_head_size
    #     attention_scores = attention_scores + attention_mask
    #
    #     # Normalize the attention scores to probabilities.
    #     attention_probs = self.softmax(attention_scores)
    #     # This is actually dropping out entire tokens to attend to, which might
    #     # seem a bit unusual, but is taken from the original Transformer paper.
    #
    #     attention_probs = self.attn_dropout(attention_probs)
    #
    #     context_layer = torch.matmul(attention_probs, value_layer)
    #     context_layer_pos = torch.matmul(attention_probs, absolute_pos_V_)
    #     context_layer_time = torch.matmul(attention_probs.unsqueeze(-2), time_matrix_emb_V_).squeeze(-2)
    #     context_layer = context_layer + context_layer_pos + context_layer_time
    #
    #     context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    #     new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    #     context_layer = context_layer.view(*new_context_layer_shape)
    #     hidden_states = self.dense(context_layer)
    #     hidden_states = self.out_dropout(hidden_states)
    #     hidden_states = self.LayerNorm(hidden_states + input_tensor)
    #
    #     return hidden_states


class ACTimeAwareTransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
            self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
            layer_norm_eps, combine_option='fixed', use_order=True, use_distance=True, two_level=True,
            rich_calibrated_combine='fixed'
    ):
        super(ACTimeAwareTransformerLayer, self).__init__()
        self.attack_attention = ACTimeAwareMultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,
            use_order=use_order, use_distance=use_distance
        )
        # self.multi_head_attention = ACTimeAwareMultiHeadAttention(
        #     n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        # )
        self.two_level = two_level
        self.rich_calibrated_combine = rich_calibrated_combine
        if self.rich_calibrated_combine == 'trainable':
            self.rich_calibrated_combine_ratio = torch.nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.combine_option = combine_option
        if self.combine_option == 'gate':
            self.gate = torch.nn.Linear(hidden_size, 50)
        self.combine_ratio = 0.5
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)
        self.anneal_step = 0

    def combine_attention(self, origin_attention, calibrated_attention, query):
        if self.combine_option == 'fixed':
            combined_attention = torch.softmax(origin_attention + self.combine_ratio * calibrated_attention, dim=-1)
        elif self.combine_option == 'gate':
            combine_gate = torch.sigmoid(self.gate(query)).unsqueeze(1)
            combined_attention = combine_gate * origin_attention + (1 - combine_gate) * calibrated_attention
        elif self.combine_option == 'annealing':
            adjust_rate = np.exp(-self.anneal_step / 100000)
            self.anneal_step += 1
            combined_attention = adjust_rate * origin_attention + (1 - adjust_rate) * calibrated_attention

        else:
            raise KeyError
        return combined_attention

    def forward(self, hidden_states, attention_mask,
                absolute_pos_K, absolute_pos_V, time_matrix_emb_K, time_matrix_emb_V,
                return_attention_prob=False):
        mixed_query, mixed_key, value, after_rich_attn_probs, before_rich_attention_probs = \
            self.attack_attention.cal_origin_qkv(hidden_states, attention_mask, absolute_pos_K,
                                                 absolute_pos_V, time_matrix_emb_K, time_matrix_emb_V)
        if self.two_level:
            origin_attn_probs = after_rich_attn_probs
        else:
            origin_attn_probs = before_rich_attention_probs
        attack_mask = self.attack_attention.cal_attack_mask(mixed_query, mixed_key, attention_mask)
        noise = torch.randn(attack_mask.shape).to(attack_mask.device)
        attacked_attention_prob = origin_attn_probs * attack_mask + noise * (1 - attack_mask)

        calibrated_attention_prob = origin_attn_probs * torch.exp(1 - attack_mask)

        combined_attention_prob = self.combine_attention(origin_attn_probs,
                                                         calibrated_attention_prob,
                                                         mixed_query)

        if not self.two_level:
            if self.rich_calibrated_combine == 'fixed':
                combined_attention_prob = (combined_attention_prob + after_rich_attn_probs) / 2
            elif self.rich_calibrated_combine == 'trainable':
                combined_attention_prob = self.rich_calibrated_combine_ratio * combined_attention_prob + \
                                          (1 - self.rich_calibrated_combine_ratio) * after_rich_attn_probs
            else:
                raise KeyError

        attacked_attention_output = self.attack_attention.cal_adjusted_outputs(attacked_attention_prob,
                                                                               hidden_states,
                                                                               value)

        calibrated_attention_output = self.attack_attention.cal_adjusted_outputs(combined_attention_prob,
                                                                                 hidden_states,
                                                                                 value)

        attacked_feedforward_output = self.feed_forward(attacked_attention_output)
        calibrated_feedforward_output = self.feed_forward(calibrated_attention_output)
        if return_attention_prob:
            return attacked_feedforward_output, calibrated_feedforward_output, attack_mask, combined_attention_prob

        return attacked_feedforward_output, calibrated_feedforward_output, attack_mask


class ACTimeAwareTransformerEncoder(nn.Module):
    r""" One TransformerEncoder consists of several TransformerLayers.

    Args:
        n_layers(num): num of transformer layers in transformer encoder. Default: 2
        n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        hidden_size(num): the input and output hidden size. Default: 64
        inner_size(num): the dimensionality in feed-forward layer. Default: 256
        hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(
            self,
            n_layers=2,
            n_heads=2,
            hidden_size=64,
            inner_size=256,
            hidden_dropout_prob=0.5,
            attn_dropout_prob=0.5,
            hidden_act='gelu',
            layer_norm_eps=1e-12,
            combine_option='fixed',
            use_order=True,
            use_distance=True,
            two_level=True,
            rich_calibrated_combine='fixed'
    ):

        super(ACTimeAwareTransformerEncoder, self).__init__()
        layer = ACTimeAwareTransformerLayer(
            n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps,
            combine_option, use_order=use_order, use_distance=use_distance, two_level=two_level,
            rich_calibrated_combine=rich_calibrated_combine
        )
        # layer = TimeAwareTransformerLayer(
        #     n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps
        # )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states,
                attention_mask,
                absolute_pos_K,
                absolute_pos_V,
                time_matrix_emb_K,
                time_matrix_emb_V,
                output_all_encoded_layers=True,
                return_attention_prob=False):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        attacked_hidden_states = None
        calibrated_hidden_states = None
        all_attack_masks = []
        all_attention_prob = [] if return_attention_prob else None
        for layer_module in self.layer:
            if return_attention_prob:
                attacked_hidden_states, calibrated_hidden_states, attack_mask, combined_attention_prob = \
                    layer_module(hidden_states, attention_mask, absolute_pos_K,
                                 absolute_pos_V, time_matrix_emb_K, time_matrix_emb_V)
            else:
                attacked_hidden_states, calibrated_hidden_states, attack_mask = \
                    layer_module(hidden_states, attention_mask, absolute_pos_K,
                                 absolute_pos_V, time_matrix_emb_K, time_matrix_emb_V)
            # hidden_states = layer_module(hidden_states, attention_mask, absolute_pos_K,
            #                              absolute_pos_V, time_matrix_emb_K, time_matrix_emb_V)
            hidden_states = calibrated_hidden_states
            all_attack_masks.append(attack_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append((attacked_hidden_states, calibrated_hidden_states))
            if return_attention_prob:
                all_attention_prob.append(combined_attention_prob)
        if not output_all_encoded_layers:
            all_encoder_layers.append((attacked_hidden_states, calibrated_hidden_states))
        if return_attention_prob:
            return all_encoder_layers, all_attack_masks, all_attention_prob
        return all_encoder_layers, all_attack_masks
        # if not output_all_encoded_layers:
        #     all_encoder_layers.append(hidden_states)
        # return all_encoder_layers

    def fake_forward(self, hidden_states, attention_mask,
                output_all_encoded_layers=True, return_attention_prob=False):
        all_encoder_layers = []
        attacked_hidden_states = None
        calibrated_hidden_states = None
        all_attack_masks = []
        all_attention_prob = [] if return_attention_prob else None
        for layer_module in self.layer:
            if return_attention_prob:
                attacked_hidden_states, calibrated_hidden_states, attack_mask, combined_attention_prob = \
                    layer_module(hidden_states, attention_mask, return_attention_prob)
            else:
                attacked_hidden_states, calibrated_hidden_states, attack_mask = \
                    layer_module(hidden_states, attention_mask, return_attention_prob)
            # print(attack_mask.shape)
            hidden_states = calibrated_hidden_states
            all_attack_masks.append(attack_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append((attacked_hidden_states, calibrated_hidden_states))
            if return_attention_prob:
                all_attention_prob.append(combined_attention_prob)
        if not output_all_encoded_layers:
            all_encoder_layers.append((attacked_hidden_states, calibrated_hidden_states))
        if return_attention_prob:
            return all_encoder_layers, all_attack_masks, all_attention_prob
        return all_encoder_layers, all_attack_masks