import math
import torch
import torch.nn as nn
from torch.nn import functional as F

INF = 1e10

# torch.matmul can't do (4, 3, 2) @ (4, 2) -> (4, 3)
def matmul(x, y):
    if x.dim() == y.dim():
        return x @ y
    if x.dim() == y.dim() - 1:
        return (x.unsqueeze(-2) @ y).squeeze(-2)
    return (x @ y.unsqueeze(-2)).squeeze(-2)

def positional_encodings_like(x, t=None):
    if t is None:
        positions = torch.arange(0, x.size(1)).float()
        if x.is_cuda:
           positions = positions.cuda(x.get_device())
    else:
        positions = t
    encodings = torch.zeros(*x.size()[1:])
    if x.is_cuda:
        encodings = encodings.cuda(x.get_device())


    for channel in range(x.size(-1)):
        if channel % 2 == 0:
            encodings[:, channel] = torch.sin(
                positions / 10000 ** (channel / x.size(2)))
        else:
            encodings[:, channel] = torch.cos(
                positions / 10000 ** ((channel - 1) / x.size(2)))
    return encodings

class Attention(nn.Module):

    def __init__(self, d_key, drop_ratio, causal):
        super(Attention).__init__()
        self.scale = math.sqrt(d_key)
        self.dropout = nn.Dropout(drop_ratio)
        self.causal = causal

    def forward(self, query, key, value):
        dot_products = matmul(query, key.transpose(1, 2))
        if query.dim() == 3 and (self is None or self.causal):
            tri = torch.ones(key.size(1), key.size(1)).triu(1) * INF
            if key.is_cuda:
                tri = tri.cuda(key.get_device())
            dot_products.data.sub_(tri.unsqueeze(0))
        A = dot_products / self.scale
        A_softmax = F.softmax(A,dim=-1)
        out = matmul(self.dropout(A_softmax),value)
        return out, A, A_softmax

class MultiHead(nn.Module):

    def __init__(self, d_key, d_value, n_heads, drop_ratio, causal=False):
        super(MultiHead).__init__()
        self.attention = Attention(d_key, drop_ratio, causal=causal)
        self.wq = nn.Linear(d_key, d_key, bias=False)
        self.wk = nn.Linear(d_key, d_key, bias=False)
        self.wv = nn.Linear(d_value, d_value, bias=False)
        self.wo = nn.Linear(d_value, d_key, bias=False)
        self.n_heads = n_heads

        self.A = None
        self.A_softmax = None

    def forward(self, query, key, value):
        query, key, value = self.wq(query), self.wk(key), self.wv(value)
        query, key, value = (
            x.chunk(self.n_heads, -1) for x in (query, key, value))
        applyA_list = []
        for q, k, v in zip(query, key, value):
            applyA,_,_ = self.attention(q,k,v)
            applyA_list.append(applyA)
        out = self.wo(torch.cat(applyA_list, -1))
        return out

    def A_forward(self, query, key, value):
        query, key, value = self.wq(query), self.wk(key), self.wv(value)
        query, key, value = (
            x.chunk(self.n_heads, -1) for x in (query, key, value))
        applyA_list = []
        A_list = []
        A_softmax_list = []
        for q, k, v in zip(query, key, value):
            applyA ,A, A_softmax = self.attention(q,k,v)
            applyA_list.append(applyA)
            A_list.append(A)
            A_softmax_list.append(A_softmax)
        out = self.wo(torch.cat(applyA_list, -1))
        self.A = torch.sum(torch.stack(A_list), 0)
        self.A_softmax = torch.sum(torch.stack(A_softmax_list), 0)
        return out

class SCDM_Attention(nn.Module):

    def __init__(self, video_dim, sent_dim, hidden_dim=None):
        super(SCDM_Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = video_dim
        self.W_s = nn.Linear(sent_dim, hidden_dim, bias=False)
        self.W_a = nn.Linear(video_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, video_feat, sent_feat):
        B, T, D_v = video_feat.size()
        B, N, D_s = sent_feat.size()
        W_s_sent_feat = self.W_s(sent_feat)
        W_a_video_feat = self.W_a(video_feat)
        P = []
        for n in range(N):
            P_n = self.w(torch.tanh(W_s_sent_feat[:, n, :].unsqueeze(dim=1) + W_a_video_feat))
            P.append(P_n)
        P_N = torch.softmax(torch.cat(P, dim=2), dim=-1)  # [B, T, N]
        C = torch.bmm(P_N, sent_feat)  # [B, T, D_s]

        return C

def masked_softmax(vec, mask, dim=1, epsilon=1e-4):
    exps = torch.exp(vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    return (masked_exps/masked_sums)

def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type_as(inputs)
    if mask.dim() == inputs.dim()-1:
        mask = mask.unsqueeze(-1).expand(-1,-1, inputs.size()[-1])
    return inputs * mask + mask_value * (1.0-mask)








