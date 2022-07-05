import torch
import torch.nn as nn
from ..networks.RNN import BiLSTM

def select_activation(name):
    name = name.lower()
    activation = nn.ReLU
    if name == 'relu':
        activation = nn.ReLU
    elif name == 'tanh':
        activation = nn.Tanh
    elif name == 'sigmoid':
        activation = nn.Sigmoid

    return activation

def select_cross(name):
    name = name.lower()
    cross = VideoTextConcat
    if name in ['concat']:
        cross = VideoTextConcat

    return VideoTextConcat

def select_temporal(name):
    name = name.lower()
    if name in ['lstm']:
        temporal = LSTMTemporal
    else:
        temporal = NoTemporal

    return temporal

def select_predict(name):
    name = name.lower()
    predict = TwoLayerdMLP
    # if name in ['mlp']:
    #     predict = TwoLayerdMLP

    return predict

class VideoTextConcat(nn.Module):
    def __init__(self, cross):
        super(VideoTextConcat, self).__init__()
        self.output_dim = cross['video_dim'] + cross['query_dim']

    def forward(self, video_feat, query_feat):
        B, T, D_v = video_feat.size()

        assert query_feat.dim() == 2 or query_feat.dim() == 3
        if query_feat.dim() == 2:
            query_feat = query_feat.unsqueeze(1).expand(B, T, -1)
        elif query_feat.dim() == 3 and query_feat.size()[1] == 1:
            query_feat = query_feat.expand(B, T, -1)
        # elif query_feat.dim() == 3 and query_feat.size()[1] != T:

        concat_feat = torch.cat([video_feat, query_feat], dim=2)
        return concat_feat

class NoTemporal(nn.Module):
    def __init__(self, temporal):
        super(NoTemporal, self).__init__()
        self.output_dim = temporal['input_dim']

    def forward(self, cross_feat):
        return cross_feat

class LSTMTemporal(nn.Module):
    def __init__(self, temporal):
        super(LSTMTemporal, self).__init__()
        self.lstm = BiLSTM(
            temporal['input_dim'],
            temporal['hidden_dim'],
            temporal['layers'],
            temporal['dropout']
        )
        self.output_dim = temporal['hidden_dim'] * 2

    def forward(self, input, *args):
        output, _, _ = self.lstm(input)
        return output

class TwoLayerdMLP(nn.Module):
    def __init__(self, predict):
        super(TwoLayerdMLP, self).__init__()
        self.activation = select_activation(predict['activation'])
        self.predict = nn.Sequential(
            nn.Linear(predict['input_dim'], predict['hidden_dim']),
            self.activation(),
            nn.Linear(predict['hidden_dim'], 1)
        )

    def forward(self, input, *args):
        output = self.predict(input).squeeze(dim=2)
        return output  # [B, T]

class VideoTextSemanticMatch(nn.Module):
    def __init__(self, cross, temporal, predict):
        super(VideoTextSemanticMatch, self).__init__()

        self.cross = select_cross(cross['name'])(cross)

        temporal['input_dim'] = self.cross.output_dim
        self.temporal = select_temporal(temporal['name'])(temporal)

        predict['input_dim'] = self.temporal.output_dim
        self.predict = select_predict(predict['name'])(predict)
        # self.activ = nn.Softmax(dim=1)

        self.temporal_dim = self.temporal.output_dim

    def forward(self, video_feat, query_feat, video_mask):
        cross_feat = self.cross(video_feat, query_feat)
        temporal_feat = self.temporal(cross_feat)
        pred_score = self.predict(temporal_feat, query_feat)


        return pred_score, temporal_feat


