import copy
import torch

from torch import nn, Tensor
from typing import Optional

from utils import _get_activation_fn
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Inherent_Layer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        # Attention Layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.modal_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # normalization & Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, ref,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        # query-representation attention
        tgt2, multi_attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(ref, pos),
                                   value=ref, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # query-query attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2, self_attn = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # query-representation attention
        tgt2, multi_attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm4(tgt)
        return tgt, self_attn, multi_attn

    def forward_pre(self, tgt, memory, ref, pos_ref,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,   # pos : structure encoding
                    query_pos: Optional[Tensor] = None):

        # multimodal attention
        tgt2 = self.norm1(tgt)
        tgt2, multi_attn = self.modal_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(ref, pos_ref),
                                   value=ref, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)

        # query-query attention
        tgt2 = self.norm2(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2, self_attn = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)

        # query-representation attention
        tgt2 = self.norm3(tgt)
        tgt2, multi_attn = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout3(tgt2)

        # FFN
        tgt2 = self.norm4(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout4(tgt2)
        return tgt, self_attn, multi_attn

    def forward(self, tgt, memory, ref, pos_ref,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, ref, pos_ref, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class Transformer_block(nn.Module):

    def __init__(self, Transformer_block, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(Transformer_block, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, ref, pos_ref,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []
        inter_self = []
        inter_multi = []

        for layer in self.layers:
            output, self_attn, multi_attn = layer(output, memory, ref, pos_ref, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                inter_self.append(self_attn)
                inter_multi.append(multi_attn)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(inter_self), torch.stack(inter_multi)

        return output, self_attn, multi_attn


class Transformer_event(nn.Module):
    def __init__(self, num_points, d_model=256, nhead=8, num_decoder_layer=6, dim_feedforward=1024,
                 dropout=0.1, activation="relu", normalize_before=True):
        super(Transformer_event, self).__init__()
        # Dim
        self.d_model = d_model
        # number of head
        self.nhead = nhead
        # structure encoding
        self.structure_encoding = nn.Parameter(torch.randn(1, num_points, d_model))
        # landmark query
        self.landmark_query = nn.Parameter(torch.randn(1, num_points, d_model))

        SLPT_Inherent_Layer = Inherent_Layer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.Transformer_block = Transformer_block(SLPT_Inherent_Layer, num_decoder_layer, decoder_norm, return_intermediate=True)

        self._reset_parameters()


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, ref, pos_ref):
        bs, num_feat, len_feat = src.size()

        structure_encoding = self.structure_encoding.repeat(bs, 1, 1).permute(1, 0, 2)
        landmark_query = self.landmark_query.repeat(bs, 1, 1).permute(1, 0, 2)

        src = src.permute(1, 0, 2)
        ref = ref.permute(1, 0, 2)

        tgt = torch.zeros_like(landmark_query)
        tgt, self_attn, multi_attn = self.Transformer_block(tgt, src, ref, pos_ref, pos=structure_encoding, query_pos=landmark_query)

        return tgt.permute(2, 0, 1, 3), self_attn, multi_attn