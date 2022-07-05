import torch
import torch.nn as nn

def word_feat_from_idx(sent_feat, inds):
    word_dim = sent_feat.size()[-1]
    feats = []
    for i in range(inds.size()[-1]):
        idxs = inds[:, :, i].unsqueeze(2).expand(-1, -1, word_dim)
        feats.append(torch.gather(sent_feat, dim=1, index=idxs))
    return feats

class GraphModeling_Triplet(nn.Module):
    def __init__(self, input_dim, hidden_dim, rl_connect):
        super(GraphModeling_Triplet, self).__init__()
        self.span_embed = SpanEmbedding()
        self.graph_model = RelationEmbedding(
            input_dim,
            hidden_dim,
            connect_type=rl_connect
        )
    def forward(self, word_encoding, obs, rls):
        object_feat_list = word_feat_from_idx(word_encoding, obs)
        object_embed = self.span_embed(object_feat_list[0], object_feat_list[1:])
        rl_feat, ob_feat, sub_feat = word_feat_from_idx(word_encoding, rls)
        triplet_embed = self.graph_model(rl_feat, ob_feat, sub_feat)
        triplet_embed = torch.cat([object_embed, triplet_embed], dim=1)

        return triplet_embed

class SpanEmbedding(nn.Module):
    def __init__(self):
        super(SpanEmbedding, self).__init__()

    def forward(self, head_feat, modifier_feats):
        return head_feat

class RelationEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, connect_type='hadamard product'):
        super(RelationEmbedding, self).__init__()
        self.message_passing = TriLinear(input_dim, hidden_dim, connect_type)

    def forward(self, edge_feat, object_feat, subject_feat):
        return self.message_passing(edge_feat, object_feat, subject_feat)

class TriLinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, connect_type='hadamard product'):
        super(TriLinear,self).__init__()
        self.wr = nn.Linear(input_dim, hidden_dim)
        self.wo = nn.Linear(input_dim, hidden_dim)
        self.ws = nn.Linear(input_dim, hidden_dim)

        self.connect_type= connect_type

        connected_dim = hidden_dim if connect_type=='hadamard product' else hidden_dim * 3
        self.we = nn.Linear(connected_dim,input_dim)
        # self.norm = nn.LayerNorm(input_dim)

    def forward(self, rl_feat, ob_feat, sub_feat):
        rl_feat = self.wr(rl_feat)
        ob_feat = self.wo(ob_feat)
        sub_feat = self.ws(sub_feat)

        if self.connect_type == 'hadamard product':
            atten_f = self.we(rl_feat * ob_feat * sub_feat)
        elif self.connect_type == 'cat':
            atten_f = self.we(torch.cat([rl_feat, ob_feat, sub_feat], dim=-1))

        atten_f = torch.relu(atten_f)

        return rl_feat + atten_f