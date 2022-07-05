import torch
import torch.nn as nn
from ..networks.RNN import BiLSTM

def select_sent_encoder(name, logger):
    if name.lower() in ['rnn', 'r']:
        sent_encoder = RNNEncoder
    else:
        logger.error('error sentence encoder name:', name,
                     'Must in \'rnn\', ')
    return sent_encoder

class RNNEncoder(nn.Module):
    def __init__(self, sent_seq_set, loggger, *args):
        super(RNNEncoder, self).__init__()
        input_dim = sent_seq_set['input_dim']
        hidden_dim = sent_seq_set['rnn_hidden_dim']
        n_layers = sent_seq_set['rnn_layers']
        drop_out = sent_seq_set['drop_out']
        self.drop_out = drop_out

        self.word_embed = nn.Linear(input_dim, input_dim)
        self.rnn_cell = BiLSTM(input_dim,
                               hidden_dim,
                               n_layers,
                               drop_out)
        self.textual_dim = hidden_dim * 2
    def forward(self, input):
        word_embedding = self.word_embed(input)
        word_encoding, hn, cn = self.rnn_cell(word_embedding)
        sent_embedding = torch.cat((hn[-2, :, :], hn[-1, :, :]), -1)
        return word_encoding, sent_embedding