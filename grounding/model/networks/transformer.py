import torch
import torch.nn as nn
from torch.nn import functional as F

from .attention import MultiHead

class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class ResidualBlock(nn.Module):

    def __init__(self, layer, d_model, drop_ratio):
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(drop_ratio)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, *x):
        # output = x[0] + self.dropout(self.layer(*x))
        # norm = self.layernorm(output.permute(0,2,1))
        # return norm.permute(0,2,1)
        return self.layernorm(x[0] + self.dropout(self.layer(*x)))

class FeedForward(nn.Module):

    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_hidden, n_heads, drop_ratio):
        super(DecoderLayer, self).__init__()
        self.selfattn = ResidualBlock(
            MultiHead(d_model, d_model, n_heads, drop_ratio),
            d_model, drop_ratio)
        self.attention = ResidualBlock(
            MultiHead(d_model, d_model, n_heads, drop_ratio),
            d_model, drop_ratio)
        self.feedforward = ResidualBlock(FeedForward(d_model, d_hidden),
                                         d_model, drop_ratio)

    def forward(self, x, encoding):
        x = self.selfattn(x, x, x)
        # out = self.feedforward(self.attention(x, encoding, encoding))
        # return out, self.attention.layer.A, self.attention.layer.A_softmax
        return self.feedforward(self.attention(x, encoding, encoding))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_hidden, n_heads, drop_ratio):
        super(EncoderLayer, self).__init__()
        self.selfattn = ResidualBlock(
            MultiHead(d_model, d_model, n_heads, drop_ratio),
            d_model, drop_ratio)
        self.feedforward = ResidualBlock(FeedForward(d_model, d_hidden),
                                         d_model, drop_ratio)

    def forward(self, x):
        return self.feedforward(self.selfattn(x, x, x))


class MHAttLayer(nn.Module):
    def __init__(self, d_model, d_hidden, n_heads, drop_ratio):
        super(MHAttLayer, self).__init__()
        self.selfattn = ResidualBlock(
            MultiHead(d_model, d_model, n_heads, drop_ratio),
            d_model, drop_ratio)
        self.feedforward = ResidualBlock(FeedForward(d_model, d_hidden),
                                         d_model, drop_ratio)

    def forward(self, q, k, v):
        return self.feedforward(self.selfattn(q, k, v))

class MHAttLayer1(nn.Module):
    def __init__(self, d_model, d_hidden, nhead, dropout):
        super(MHAttLayer1, self).__init__()
        self.mh_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_hidden, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, q, k, v):
        """

        :param q: [B, T, E]
        :param k: [B, N, E]
        :param v: [B, N, E]
        :return:
        """
        q = q.permute(1,0,2)
        k = k.permute(1,0,2)
        v = v.permute(1,0,2)
        att = self.mh_attn(q, k, v)[0]
        att = q + self.dropout1(att)
        att = self.norm1(att)
        ffn = self.linear2(self.dropout(self.activation(self.linear1(att))))
        output = att + self.dropout2(ffn)
        output = self.norm2(output)
        return output.permute(1,0,2)