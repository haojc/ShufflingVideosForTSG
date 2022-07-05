import torch
import torch.nn as nn
from ..networks.RNN import BiLSTM
from ..networks.attention import SCDM_Attention

def select_video_encoder(name, logger):
    name = name.lower()
    if name in ['rnn', 'r']:
        video_encoder = RNNEncoder
    elif name in ['query_aware_encoder', 'qae', 'qave']:
        video_encoder = QueryAwareEncoder
    else:
        logger.error('error video encoder name:', name,
                     'Must in \'rnn\', \'qae\', ')
    return video_encoder

class RNNEncoder(nn.Module):
    ''' pure viusal encoder, no query information introduced '''

    def __init__(self, video_seq_set, logger, *args):
        super(RNNEncoder, self).__init__()

        video_dim = video_seq_set['input_dim']
        hidden_dim = video_seq_set['rnn_hidden_dim']
        n_layers = video_seq_set['rnn_layers']
        drop_out = video_seq_set['drop_out']

        self.rnn_cell = BiLSTM(video_dim,
                               hidden_dim,
                               n_layers,
                               drop_out)
        self.visual_dim = hidden_dim * 2
        self.video_layernorm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, input, *args):
        video_encoding, hn, cn = self.rnn_cell(input)
        global_embedding = torch.cat((hn[-2, :, :], hn[-1, :, :]), -1)
        video_encoding = self.video_layernorm(video_encoding)
        return video_encoding

class rnn_recalibration_layer(nn.Module):
    def __init__(self, input_dim, sent_dim, hidden_dim, n_layers, ca_activ, drop_out, logger):
        super(rnn_recalibration_layer, self).__init__()

        self.ca_activ =ca_activ

        self.rnn_cell = BiLSTM(
            input_dim,
            hidden_dim,
            n_layers,
            drop_out
        )
        self.visual_dim = hidden_dim * 2

        self.attention = SCDM_Attention(
            self.visual_dim,
            sent_dim
        )
        self.sent_linear = nn.Linear(sent_dim, self.visual_dim)

    def forward(self, video_feat, word_feat):
        rnn_output, hn, cn = self.rnn_cell(video_feat)
        C = self.attention(rnn_output, word_feat)

        channel_attn = self.sent_linear(C)
        if self.ca_activ in ['sigmoid']:
            channel_attn = torch.sigmoid(channel_attn)
        elif self.ca_activ in ['relu']:
            channel_attn = torch.relu(channel_attn)
        elif self.ca_activ in ['tanh']:
            channel_attn = torch.tanh(channel_attn)
        gated_video_feat = rnn_output * channel_attn

        return gated_video_feat

class QueryAwareEncoder(nn.Module):
    def __init__(self, video_seq_set, logger, *args):
        super(QueryAwareEncoder, self).__init__()

        hidden_dim = video_seq_set['rnn_hidden_dim']
        n_layers = video_seq_set['rnn_layers']
        drop_out = video_seq_set['drop_out']
        sent_dim = video_seq_set['query_dim']
        ca_activ = 'sigmoid'
        self.nblocks = video_seq_set['nblocks']

        input_dim = video_seq_set['input_dim']
        block = rnn_recalibration_layer
        self.blocks = nn.ModuleList()
        for _ in range(self.nblocks):
            b =  block(input_dim, sent_dim, hidden_dim, n_layers, ca_activ, drop_out, logger)
            self.blocks.append(b)
            input_dim = hidden_dim * 2

        self.visual_dim = hidden_dim * 2
        self.norm = nn.LayerNorm(self.visual_dim)

    def forward(self, video_feat, query_feat, *args):
        B, T, _ = video_feat.size()

        if not isinstance(query_feat, list):
            query_list = [query_feat] * self.nblocks
        elif len(query_feat) < self.nblocks:
            query_list = query_feat + [query_feat[-1]] * (self.nblocks - len(query_feat))
        else:
            query_list = query_feat

        residual = video_feat
        for i in range(self.nblocks):
            output = self.blocks[i](residual, query_list[i])
            residual = output
        output = self.norm(output)

        return output
