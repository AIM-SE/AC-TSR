"""
TiSASRec
################################################

Reference:
    Jiacheng Li et al. "Time Interval Aware Self-Attention for Sequential Recommendation." in WSDM 2020.

Reference:
    https://github.com/JiachengLi1995/TiSASRec

"""

import numpy as np
import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.transformer_layers import ACTimeAwareTransformerEncoder
from recbole.model.loss import BPRLoss


class ACTiSASRec(SequentialRecommender):

    def __init__(self, config, dataset):
        super(ACTiSASRec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.time_span = config['time_span']
        self.timestamp = config['TIME_FIELD'] + '_list'

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        self.combine_option = config['combine_option']

        self.rich_calibrated_combine = config['rich_calibrated_combine']
        self.two_level = config['two_level']
        self.use_position_embedding = config['use_position_embedding']
        self.use_order = config['use_order']
        self.use_distance = config['use_distance']
        self.trainable_mask_loss_weight = config['trainable_mask_loss_weight']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.absolute_pos_K_embedding = nn.Embedding(self.max_seq_length, self.hidden_size, padding_idx=0)
        self.absolute_pos_V_embedding = nn.Embedding(self.max_seq_length, self.hidden_size, padding_idx=0)
        self.time_matrix_emb_K_embedding = nn.Embedding(self.time_span + 1, self.hidden_size, padding_idx=0)
        self.time_matrix_emb_V_embedding = nn.Embedding(self.time_span + 1, self.hidden_size, padding_idx=0)

        self.ti_trm_encoder = ACTimeAwareTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            combine_option=self.combine_option,
            use_order=self.use_order,
            use_distance=self.use_distance,
            two_level=self.two_level,
            rich_calibrated_combine=self.rich_calibrated_combine
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.trainable_mask_loss_weight = config['trainable_mask_loss_weight']
        if self.trainable_mask_loss_weight:
            self.mask_loss_weight = nn.Parameter(torch.FloatTensor([0.3]), requires_grad=True)
        else:
            self.mask_loss_weight = config['mask_loss_weight']

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len, time_matrix):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)

        absolute_pos_K = self.absolute_pos_K_embedding(position_ids)  # [B, L, D]
        absolute_pos_V = self.absolute_pos_V_embedding(position_ids)

        time_matrix_emb_K = self.time_matrix_emb_K_embedding(time_matrix)  # [B, L, L, D]
        time_matrix_emb_V = self.time_matrix_emb_V_embedding(time_matrix)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        absolute_pos_K = self.dropout(absolute_pos_K)
        absolute_pos_V = self.dropout(absolute_pos_V)
        time_matrix_emb_K = self.dropout(time_matrix_emb_K)
        time_matrix_emb_V = self.dropout(time_matrix_emb_V)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.ti_trm_encoder(
            input_emb,
            extended_attention_mask,
            absolute_pos_K,
            absolute_pos_V,
            time_matrix_emb_K,
            time_matrix_emb_V,
            output_all_encoded_layers=True
        )
        attacked_output, calibrated_output = trm_output[0][-1]
        all_attack_masks = trm_output[1]
        attacked_output = self.gather_indexes(attacked_output, item_seq_len - 1)
        calibrated_output = self.gather_indexes(calibrated_output, item_seq_len - 1)
        return attacked_output, calibrated_output, all_attack_masks
        # raise NotImplementedError
        # output = trm_output[-1]
        # output = self.gather_indexes(output, item_seq_len - 1)
        #
        # return output  # [B H]

    def get_time_matrix(self, time_seq):  # time_seq -> time_matrix: [B, L] -> [B, L, L]
        time_seq = time_seq

        time_matrix_i = time_seq.unsqueeze(-1).expand([-1, self.max_seq_length, self.max_seq_length])
        time_matrix_j = time_seq.unsqueeze(1).expand([-1, self.max_seq_length, self.max_seq_length])
        time_matrix = torch.abs(time_matrix_i - time_matrix_j)
        max_time_matrix = (torch.ones_like(time_matrix) * self.time_span).to(self.device)
        time_matrix = torch.where(time_matrix > self.time_span, max_time_matrix, time_matrix).int()

        return time_matrix

    def _cal_loss(self, seq_output, interaction):
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
        return loss

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        time_seq = interaction[self.timestamp]
        time_matrix = self.get_time_matrix(time_seq)

        attacked_output, calibrated_output, all_attack_masks = self.forward(item_seq, item_seq_len, time_matrix)

        attacked_loss = -self._cal_loss(attacked_output, interaction)
        calibrated_loss = self._cal_loss(calibrated_output, interaction)
        mask_penalty = []
        for attack_mask in all_attack_masks:
            mask_penalty.append(torch.norm(1 - attack_mask, p=2))
        mask_penalty = torch.mean(torch.stack(mask_penalty, dim=0))
        if self.trainable_mask_loss_weight:
            final_attacked_loss = attacked_loss + mask_penalty * self.mask_loss_weight[0]
        else:
            final_attacked_loss = attacked_loss + mask_penalty * self.mask_loss_weight

        return final_attacked_loss, calibrated_loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]

        time_seq = interaction[self.timestamp]
        time_matrix = self.get_time_matrix(time_seq)

        attacked_output, calibrated_output, _ = self.forward(item_seq, item_seq_len, time_matrix)

        test_item_emb = self.item_embedding(test_item)
        attacked_scores = torch.mul(attacked_output, test_item_emb).sum(dim=1)  # [B]
        scores = torch.mul(calibrated_output, test_item_emb).sum(dim=1)  # [B]
        return attacked_scores, scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        time_seq = interaction[self.timestamp]
        time_matrix = self.get_time_matrix(time_seq)

        attacked_output, calibrated_output, _ = self.forward(item_seq, item_seq_len, time_matrix)

        test_items_emb = self.item_embedding.weight
        attacked_scores = torch.matmul(attacked_output, test_items_emb.transpose(0, 1))  # [B]
        scores = torch.matmul(calibrated_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return attacked_scores, scores
