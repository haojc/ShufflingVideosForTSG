import torch.nn as nn
import torch

class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.GRU = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)

    def forward(self, x, h0=None):
        # Set initial states
        if h0 is None:
            h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size,
                             dtype=torch.float).cuda()  # 2 for bidirection

        # Forward propagate LSTM
        self.GRU.flatten_parameters()
        out, hn = self.GRU(x, h0)
        # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        # hn:  tensor of shape (batch_size, num_layers, hidden_size*2)

        return out, hn, 0

# Bidirectional recurrent neural network (many-to-one)
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        # self.fc = nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirection

    def forward(self, x, h0 = None, c0 = None):
        # Set initial states
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, dtype=torch.float).cuda()  # 2 for bidirection
            c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, dtype=torch.float).cuda()

        # Forward propagate LSTM
        self.lstm.flatten_parameters()
        out, (hn,cn) = self.lstm(x, (h0, c0))
        # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        # hn:  tensor of shape (batch_size, num_layers, hidden_size*2)
        # cn:  tensor of shape (batch_size, num_layers, hidden_size*2)

        # Decode the hidden state of the last time step
        # out = self.fc(out[:, -1, :])
        return out, hn, cn