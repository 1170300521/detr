import math
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['MultiHeadAttention', 'ScaledDotProductAttention']


class ScaledDotProductAttention(nn.Module):
    
    def __init__(self,
                 dropout=0,
                 head_num=8,
                 deepvit=True):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.head_att = nn.parameter.Parameter(torch.Tensor(head_num, head_num)) if deepvit \
            else None
        self.deepvit = deepvit
        self.head_num = head_num
        self.norm = nn.LayerNorm(head_num)
        self._reset_parameters()

    def _reset_parameters(self):
        if self.head_att is not None:
            nn.init.xavier_uniform_(self.head_att)

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        bh, q_len, k_len = attention.shape
        attention = attention.view(-1, self.head_num, q_len, k_len)
        att_map = attention.mean(1)
        if self.deepvit:
            attention = attention.permute(0, 2, 3, 1)
            attention = attention.matmul(self.head_att)
            attention = self.norm(attention).permute(0, 3, 1, 2).contiguous()
            att_map = attention.mean(1)
            attention = attention.view(bh, q_len, k_len)
        return attention.matmul(value), att_map


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 dropout=0,
                 deepvit=True,
                 bias=True,
                 activation=None):
        """Multi-head attention.

        :param in_features: Size of each input sample.
        :param dropout: Dropout rate of dropout layer
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)
        self.scale_dot = ScaledDotProductAttention(dropout, head_num, deepvit)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        TODO Note: Ignore query mask now, we will fixed this later
        """
        #print(query.shape)
        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)
        q, k, v = self.linear_q(query), self.linear_k(key), self.linear_v(value)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        # print(key_padding_mask.shape)
        if key_padding_mask is not None:
            # print(key_padding_mask)
            bs, seq_len = key_padding_mask.shape
            key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, self.head_num, 1)
            key_padding_mask = key_padding_mask.view(bs*self.head_num, 1, seq_len)
            # print(key_padding_mask)
        y, att_map = self.scale_dot(q, k, v, key_padding_mask)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        # print(y.shape)
        if self.activation is not None:
            y = self.activation(y)
        y = y.transpose(1, 0)
        return y, att_map

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.

        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )
