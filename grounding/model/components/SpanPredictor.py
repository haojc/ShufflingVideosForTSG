import torch
import torch.nn as nn
from ..networks.RNN import BiLSTM
from ..networks.attention import MultiHead, positional_encodings_like, mask_logits
import numpy as np

class SpanPredictor_Boundary(nn.Module):

    def __init__(self, crossmodal_dim, predictor_set, drop_out, logger):
        super(SpanPredictor_Boundary, self).__init__()
        self.crossmodal_dim = crossmodal_dim
        self.drop_out = drop_out

        if predictor_set['name'] in ['mlp', 'a']:
            self.predictor = MLP_predictor(self.crossmodal_dim, predictor_set['mlp_hidden_dim'])
        elif predictor_set['name'] in ['tied_lstm', 'b']:
            self.predictor = Tied_LSTM_predictor(self.crossmodal_dim,
                                                 predictor_set['lstm_hidden_dim'],
                                                 predictor_set['mlp_hidden_dim'],
                                                 self.drop_out)
        elif predictor_set['name'] in ['cat_tied_lstm', 'b2']:
            self.predictor = cat_Tied_LSTM_predictor(self.crossmodal_dim,
                                                     predictor_set['lstm_hidden_dim'],
                                                     predictor_set['mlp_hidden_dim'],
                                                     self.drop_out)
        elif predictor_set['name'] in ['condi_lstm', 'c']:
            self.predictor = Conditional_LSTM_predictor(self.crossmodal_dim,
                                                        predictor_set['lstm_hidden_dim'],
                                                        self.drop_out)
        elif predictor_set['name'] in ['cat_condi_lstm', 'c2']:
            self.predictor = cat_Conditional_LSTM_predictor(self.crossmodal_dim,
                                                            predictor_set['lstm_hidden_dim'],
                                                            predictor_set['mlp_hidden_dim'],
                                                            self.drop_out)
        elif predictor_set['name'] in ['self_attn', 'd']:
            self.predictor = Self_Attention_predictor(self.crossmodal_dim,
                                                      predictor_set['attention_nheads'],
                                                      predictor_set['position_encoding'],
                                                      self.drop_out)
        else:
            logger.error('error predictor name:', predictor_set['name'],
                         'Must in \'mlp\', \'tied_lstm\', \'condi_lstm\' or \'self_attn\' ')
    def forward(self, crossmodal_feat, v_mask=None):
        start_prob, end_prob = self.predictor(crossmodal_feat, v_mask)

        return start_prob, end_prob

class ConvPredictor(nn.Module):
    ''' Conv kernel as the sapn predictor '''
    def __init__(self, input_dim, kernel_size, num_kernel):
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=num_kernel,
            kernel_size= kernel_size,
            stride= 1,
            padding= 0,
        )


class MLP_predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP_predictor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.start_mlp_1 = nn.Linear(input_dim, hidden_dim)
        self.start_mlp_2 = nn.Linear(hidden_dim, 1)

        self.end_mlp_1 = nn.Linear(input_dim, hidden_dim)
        self.end_mlp_2 = nn.Linear(hidden_dim, 1)

    def forward(self, crossmodal_feat, v_mask=None):
        start_prob = self.start_mlp_2(torch.tanh(self.start_mlp_1(crossmodal_feat)))
        end_prob = self.end_mlp_2(torch.tanh(self.end_mlp_1(crossmodal_feat)))  # [batch, T, 1]
        # print(start_prob.size(),end_prob.size())
        start_prob = start_prob.squeeze(dim=2)
        end_prob = end_prob.squeeze(dim=2)

        if v_mask is not None:
            start_prob = mask_logits(start_prob, mask=v_mask)
            end_prob = mask_logits(end_prob, mask=v_mask)

        start_prob = torch.softmax(start_prob, dim=1)
        end_prob = torch.softmax(end_prob, dim=1)

        return start_prob, end_prob

class cat_Tied_LSTM_predictor(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, mlp_hidden_dim, drop_out):
        super(cat_Tied_LSTM_predictor, self).__init__()
        self.input_dim = input_dim
        self.cross_lstm_dim = lstm_hidden_dim
        self.mlp_hidden_dim = mlp_hidden_dim

        self.cross_lstm = BiLSTM(input_dim,
                                 self.cross_lstm_dim,
                                 num_layers=1,
                                 dropout=drop_out
                                 )

        mlp_input_dim = self.cross_lstm_dim*2 + input_dim
        self.start_mlp_1 = nn.Linear(mlp_input_dim, mlp_hidden_dim)
        self.start_mlp_2 = nn.Linear(mlp_hidden_dim, 1)

        self.end_mlp_1 = nn.Linear(mlp_input_dim, mlp_hidden_dim)
        self.end_mlp_2 = nn.Linear(self.mlp_hidden_dim, 1)

    def forward(self, crossmodal_feat, v_mask=None):
        crossmodal_lstm_feat, _, _ = self.cross_lstm(crossmodal_feat)  # [B, T, cross_lstm_dim]
        mlp_input_feat = torch.cat([crossmodal_lstm_feat, crossmodal_feat], dim=-1)
        start_prob = self.start_mlp_2(torch.tanh(self.start_mlp_1(mlp_input_feat)))
        end_prob = self.end_mlp_2(torch.tanh(self.end_mlp_1(mlp_input_feat)))  # [batch, T, 1]


        start_prob = start_prob.squeeze(dim=2)
        end_prob = end_prob.squeeze(dim=2)

        if v_mask is not None:
            start_prob = mask_logits(start_prob, mask=v_mask)
            end_prob = mask_logits(end_prob, mask=v_mask)
        start_prob = torch.softmax(start_prob, dim=1)
        end_prob = torch.softmax(end_prob, dim=1)

        return start_prob, end_prob

class cat_Conditional_LSTM_predictor(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, mlp_hidden_dim, drop_out):
        super(cat_Conditional_LSTM_predictor, self).__init__()
        self.crossmodal_dim = input_dim
        self.cross_lstm_dim = lstm_hidden_dim

        self.start_lstm = BiLSTM(self.crossmodal_dim,
                                 self.cross_lstm_dim,
                                 num_layers=1,
                                 dropout=drop_out
                                 )
        self.end_lstm = BiLSTM(self.cross_lstm_dim * 2,
                               self.cross_lstm_dim,
                               num_layers=1,
                               dropout=drop_out
                               )

        self.start_mlp_1 = nn.Linear(self.cross_lstm_dim * 2 + self.crossmodal_dim, mlp_hidden_dim)
        self.start_mlp_2 = nn.Linear(mlp_hidden_dim, 1)

        self.end_mlp_1 = nn.Linear(self.cross_lstm_dim * 2 + self.crossmodal_dim, mlp_hidden_dim)
        self.end_mlp_2 = nn.Linear(mlp_hidden_dim, 1)

    def forward(self, crossmodal_feat, v_mask=None):
        start_feat, _, _ = self.start_lstm(crossmodal_feat)
        end_feat, _, _ = self.end_lstm(start_feat)

        start_feat = torch.cat([start_feat, crossmodal_feat], dim=-1)
        end_feat = torch.cat([end_feat, crossmodal_feat], dim=-1)

        start_prob = self.start_mlp_2(torch.tanh(self.start_mlp_1(start_feat)))
        end_prob = self.end_mlp_2(torch.tanh(self.end_mlp_1(end_feat)))  # [batch, T, 1]

        start_prob = start_prob.squeeze(dim=2)
        end_prob = end_prob.squeeze(dim=2)

        if v_mask is not None:
            start_prob = mask_logits(start_prob, mask=v_mask)
            end_prob = mask_logits(end_prob, mask=v_mask)
        start_prob = torch.softmax(start_prob, dim=1)
        end_prob = torch.softmax(end_prob, dim=1)

        return start_prob, end_prob

class Tied_LSTM_predictor(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, mlp_hidden_dim, drop_out):
        super(Tied_LSTM_predictor, self).__init__()
        self.input_dim = input_dim
        self.cross_lstm_dim = lstm_hidden_dim
        self.mlp_hidden_dim = mlp_hidden_dim

        self.cross_lstm = BiLSTM(input_dim,
                                 self.cross_lstm_dim,
                                 num_layers=1,
                                 dropout=drop_out
                                 )

        self.start_mlp_1 = nn.Linear(self.cross_lstm_dim * 2, mlp_hidden_dim)
        self.start_mlp_2 = nn.Linear(mlp_hidden_dim, 1)

        self.end_mlp_1 = nn.Linear(self.cross_lstm_dim * 2, mlp_hidden_dim)
        self.end_mlp_2 = nn.Linear(self.mlp_hidden_dim, 1)

    def forward(self, crossmodal_feat, v_mask):
        crossmodal_lstm_feat, _, _ = self.cross_lstm(crossmodal_feat)  # [B, T, cross_lstm_dim]
        # print(crossmodal_lstm_feat.size())

        start_prob = self.start_mlp_2(torch.tanh(self.start_mlp_1(crossmodal_lstm_feat)))
        end_prob = self.end_mlp_2(torch.tanh(self.end_mlp_1(crossmodal_lstm_feat)))  # [batch, T, 1]
        # print(start_prob.size(),end_prob.size())

        start_prob = start_prob.squeeze(dim=2)
        end_prob = end_prob.squeeze(dim=2)

        if v_mask is not None:
            start_prob = mask_logits(start_prob, mask=v_mask)
            end_prob = mask_logits(end_prob, mask=v_mask)

        start_prob = torch.softmax(start_prob, dim=1)
        end_prob = torch.softmax(end_prob, dim=1)

        return start_prob, end_prob


class Conditional_LSTM_predictor(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, drop_out):
        super(Conditional_LSTM_predictor, self).__init__()
        self.crossmodal_dim = input_dim
        self.cross_lstm_dim = lstm_hidden_dim

        self.start_lstm = BiLSTM(self.crossmodal_dim,
                                 self.cross_lstm_dim,
                                 num_layers=1,
                                 dropout=drop_out
                                 )
        self.end_lstm = BiLSTM(self.cross_lstm_dim * 2,
                               self.cross_lstm_dim,
                               num_layers=1,
                               dropout=drop_out
                               )

        self.start_fc = nn.Linear(self.cross_lstm_dim * 2, 1)
        self.end_fc = nn.Linear(self.cross_lstm_dim * 2, 1)

    def forward(self, crossmodal_feat, v_mask):
        start_feat, _, _ = self.start_lstm(crossmodal_feat)
        end_feat, _, _ = self.end_lstm(start_feat)
        start_prob = self.start_fc(start_feat).squeeze(dim=2)
        end_prob = self.end_fc(end_feat).squeeze(dim=2)

        if v_mask is not None:
            start_prob = mask_logits(start_prob, mask=v_mask)
            end_prob = mask_logits(end_prob, mask=v_mask)

        start_prob = torch.softmax(start_prob, dim=1)
        end_prob = torch.softmax(end_prob, dim=1)

        return start_prob, end_prob

class Self_Attention_predictor(nn.Module):
    def __init__(self, input_dim, n_heads, position_encoding, drop_out):
        super(Self_Attention_predictor, self).__init__()
        self.crossmodal_dim = input_dim
        self.position_encoding = position_encoding

        self.start_selfattn = MultiHead(input_dim, input_dim, n_heads, drop_out)
        self.end_selfattn = MultiHead(input_dim, input_dim, n_heads, drop_out)

        self.start_fc = nn.Linear(input_dim, 1)
        self.end_fc = nn.Linear(input_dim, 1)

    def forward(self, crossmodal_feat):
        if self.position_encoding:
            crossmodal_feat = crossmodal_feat + positional_encodings_like(crossmodal_feat)

        start_feat = self.start_selfattn(crossmodal_feat, crossmodal_feat, crossmodal_feat)
        end_feat = self.end_selfattn(crossmodal_feat, crossmodal_feat, crossmodal_feat)

        start_prob = torch.softmax(self.start_fc(start_feat).squeeze(dim=2), dim=1)
        end_prob = torch.softmax(self.end_fc(end_feat).squeeze(dim=2), dim=1)

        return start_prob, end_prob


"""
content predictor
output: start_prob, end_prob, content_prob
"""

class MLP_content_predictor(nn.Module):
    """
    """
    def __init__(self, input_dim, hidden_dim):
        super(MLP_content_predictor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.start_mlp_1 = nn.Linear(input_dim, hidden_dim)
        self.start_mlp_2 = nn.Linear(hidden_dim, 1)

        self.end_mlp_1 = nn.Linear(input_dim, hidden_dim)
        self.end_mlp_2 = nn.Linear(hidden_dim, 1)

        self.content_mlp_1 = nn.Linear(input_dim, hidden_dim)
        self.content_mlp_2 = nn.Linear(hidden_dim, 1)

    def forward(self, crossmodal_feat):
        start_prob = self.start_mlp_2(torch.tanh(self.start_mlp_1(crossmodal_feat)))
        end_prob = self.end_mlp_2(torch.tanh(self.end_mlp_1(crossmodal_feat)))  # [batch, T, 1]
        content_prob = self.content_mlp_2(torch.tanh(self.content_mlp_1(crossmodal_feat)))  # [batch, T, 1]

        # print(start_prob.size(),end_prob.size())
        start_prob = torch.softmax(start_prob.squeeze(dim=2), dim=1)
        end_prob = torch.softmax(end_prob.squeeze(dim=2), dim=1)
        content_prob = torch.softmax(content_prob.squeeze(dim=2), dim=1)

        return start_prob, end_prob, content_prob

class Tied_LSTM_content_predictor(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, mlp_hidden_dim, drop_out):
        super(Tied_LSTM_content_predictor, self).__init__()
        self.input_dim = input_dim
        self.cross_lstm_dim = lstm_hidden_dim
        self.mlp_hidden_dim = mlp_hidden_dim

        self.cross_lstm = BiLSTM(input_dim,
                                 self.cross_lstm_dim,
                                 num_layers=1,
                                 dropout=drop_out
                                 )

        self.start_mlp_1 = nn.Linear(self.cross_lstm_dim * 2, mlp_hidden_dim)
        self.start_mlp_2 = nn.Linear(mlp_hidden_dim, 1)

        self.end_mlp_1 = nn.Linear(self.cross_lstm_dim * 2, mlp_hidden_dim)
        self.end_mlp_2 = nn.Linear(self.mlp_hidden_dim, 1)

        self.content_mlp_1 = nn.Linear(self.cross_lstm_dim * 2, mlp_hidden_dim)
        self.content_mlp_2 = nn.Linear(self.mlp_hidden_dim, 1)

    def forward(self, crossmodal_feat):
        crossmodal_lstm_feat, _, _ = self.cross_lstm(crossmodal_feat)  # [B, T, cross_lstm_dim]
        # print(crossmodal_lstm_feat.size())

        start_prob = self.start_mlp_2(torch.tanh(self.start_mlp_1(crossmodal_lstm_feat)))
        end_prob = self.end_mlp_2(torch.tanh(self.end_mlp_1(crossmodal_lstm_feat)))  # [batch, T, 1]
        content_prob = self.content_mlp_2(torch.tanh(self.content_mlp_1(crossmodal_lstm_feat)))  # [batch, T, 1]

        # print(start_prob.size(),end_prob.size())
        start_prob = torch.softmax(start_prob.squeeze(dim=2), dim=1)
        end_prob = torch.softmax(end_prob.squeeze(dim=2), dim=1)
        content_prob = torch.softmax(content_prob.squeeze(dim=2), dim=1)

        return start_prob, end_prob, content_prob


class Conditional_LSTM_content_predictor(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, drop_out):
        super(Conditional_LSTM_content_predictor, self).__init__()
        self.crossmodal_dim = input_dim
        self.cross_lstm_dim = lstm_hidden_dim

        self.start_lstm = BiLSTM(self.crossmodal_dim,
                                 self.cross_lstm_dim,
                                 num_layers=1,
                                 dropout=drop_out
                                 )
        self.end_lstm = BiLSTM(self.cross_lstm_dim * 2,
                               self.cross_lstm_dim,
                               num_layers=1,
                               dropout=drop_out
                               )

        self.content_lstm = BiLSTM(self.cross_lstm_dim * 2,
                               self.cross_lstm_dim,
                               num_layers=1,
                               dropout=drop_out
                               )

        self.start_fc = nn.Linear(self.cross_lstm_dim * 2, 1)
        self.end_fc = nn.Linear(self.cross_lstm_dim * 2, 1)
        self.content_fc = nn.Linear(self.cross_lstm_dim * 2, 1)

    def forward(self, crossmodal_feat):
        start_feat, _, _ = self.start_lstm(crossmodal_feat)
        end_feat, _, _ = self.end_lstm(start_feat)
        content_feat, _, _ = self.content_lstm(start_feat)

        start_prob = torch.softmax(self.start_fc(start_feat).squeeze(dim=2), dim=1)
        end_prob = torch.softmax(self.end_fc(end_feat).squeeze(dim=2), dim=1)
        content_prob = torch.softmax(self.content_fc(content_feat).squeeze(dim=2), dim=1)

        return start_prob, end_prob, content_prob


class start_condi_predictor(nn.Module):
    def __init__(self, video_dim, sent_dim, hidden_dim, lstm_hidden_dim, drop_out):
        super(start_condi_predictor, self).__init__()
        self.nolate = True

        start_input_dim = video_dim if self.nolate else video_dim + sent_dim
        self.start_mlp_1 = nn.Linear(start_input_dim, hidden_dim)
        self.start_mlp_2 = nn.Linear(hidden_dim, 1)

        end_input_dim = video_dim * 2 if self.nolate else video_dim*2 + sent_dim
        self.end_lstm = BiLSTM(end_input_dim,
                               lstm_hidden_dim,
                               num_layers=2,
                               dropout=drop_out
                               )
        self.end_mlp_1 = nn.Linear(lstm_hidden_dim*2, hidden_dim)
        self.end_mlp_2 = nn.Linear(hidden_dim, 1)

    def forward(self, video_feat, sent_embedding, start_timestamp):
        B, T, D = video_feat.size()

        crossmodal_feat = video_feat if self.nolate \
            else torch.cat((video_feat, sent_embedding), -1)

        start_prob = self.start_mlp_2(torch.tanh(self.start_mlp_1(crossmodal_feat)))

        # condi_feat = video_feat[start_timestamp]
        start_timestamp = start_timestamp.view(B,1,1).expand(-1,-1,D)
        condi_feat = torch.gather(video_feat, dim=1, index=start_timestamp).expand(-1,T,-1)

        crossmodal_feat = torch.cat((crossmodal_feat, condi_feat), -1)
        end_feat,_,_ = self.end_lstm(crossmodal_feat)
        end_prob = self.end_mlp_2(torch.tanh(self.end_mlp_1(end_feat)))  # [batch, T, 1]
        # print(start_prob.size(),end_prob.size())

        start_prob = torch.softmax(start_prob.squeeze(dim=2), dim=1)
        end_prob = torch.softmax(end_prob.squeeze(dim=2), dim=1)

        return start_prob, end_prob

    def inference(self, video_feat, sent_embedding):
        B, T, D = video_feat.size()

        crossmodal_feat = video_feat if self.nolate \
            else torch.cat((video_feat, sent_embedding), -1)

        start_prob = self.start_mlp_2(torch.tanh(self.start_mlp_1(crossmodal_feat)))
        start_prob = torch.softmax(start_prob.squeeze(dim=2), dim=1)

        _, s_max_idx = start_prob.max(dim=1)
        s_max_idx = s_max_idx.unsqueeze(dim=1).expand(B,D).unsqueeze(dim=1) #[B, 1, D]

        condi_feat = torch.gather(video_feat, dim=1, index=s_max_idx).expand(B,T,D)
        crossmodal_feat = torch.cat((crossmodal_feat, condi_feat),dim=-1)
        end_feat, _, _ = self.end_lstm(crossmodal_feat)
        end_prob = self.end_mlp_2(torch.tanh(self.end_mlp_1(end_feat)))  # [batch, T, 1]

        end_prob = torch.softmax(end_prob.squeeze(dim=2), dim=1)

        return start_prob, end_prob