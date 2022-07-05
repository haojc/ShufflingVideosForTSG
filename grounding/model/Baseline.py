import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.autograd import Function

from .networks.attention import *
from .components import SentenceEncoder, VideoEncoder, SpanPredictor, CrossModalInteraction
from .components.DistributionAlign import VideoTextSemanticMatch

class Baseline(nn.Module):
    def __init__(self, video_seq_set, sent_seq_set, grounding_set, matching_set, logger, drop_out):
        super(Baseline, self).__init__()

        # Default setting
        # sent_seq_set['name'] = 'rnn'
        # video_seq_set['name'] = 'qave'
        # grounding_set['cross_name'] = 'vs'
        # grounding_set['name'] = 'a'

        # Sentece Encoder
        sent_encoder = SentenceEncoder.select_sent_encoder(sent_seq_set['name'], logger)
        self.sentence_encoder = sent_encoder(
            sent_seq_set, logger
        )
        self.textual_dim = self.sentence_encoder.textual_dim

        # Video Encoder
        video_seq_set['query_dim'] = self.textual_dim
        self.query_level_in_video = 'word'
        video_encoder = VideoEncoder.select_video_encoder(video_seq_set['name'], logger)
        self.video_encoder = video_encoder(
            video_seq_set, logger
        )
        self.visual_dim = self.video_encoder.visual_dim
        self.video_if_mask = video_seq_set['mask']

        # Grounding
        self.CMI = CrossModalInteraction.select_CMI(grounding_set['cross_name'], logger)(
            self.visual_dim,
            self.textual_dim
        )
        self.cross_dim = self.CMI.cross_dim()

        self.span_predictor = SpanPredictor.SpanPredictor_Boundary(
            self.cross_dim,
            grounding_set,
            drop_out=drop_out,
            logger=logger,
        )

        # By default setting, QAVE doesn't have a temporal gating module
        # matching_set['cross']['video_dim'] = self.visual_dim
        # matching_set['cross']['query_dim'] = self.textual_dim
        # self.csmm = VideoTextSemanticMatch(
        #     matching_set['cross'],
        #     matching_set['temporal'],
        #     matching_set['predict']
        # )
        # self.matching_dim = self.csmm.temporal_dim


    def forward(self, video_feat, query_feat,
                video_mask=None, query_mask=None):
        _, N_s, D_s_s = query_feat.size()

        # Natural Language Modality
        word_feature, sent_embed = self.sentence_encoder(query_feat)

        # Video Modality
        frame_feature = self.video_encoder(video_feat, word_feature)

        # Grounding
        # Cross Modal
        cross_feat = self.CMI(frame_feature, word_feature, sent_embed)

        #Matching
        # match_prob, matching_feat = self.csmm(
        #     frame_feature, sent_embed, video_mask
        # )


        # Span Predictor
        # input_feat_to_span = match_prob.unsqueeze(dim=2) * cross_feat
        input_feat_to_span = cross_feat
        start_prob, end_prob = self.span_predictor(
            input_feat_to_span,
            v_mask=video_mask if self.video_if_mask else None,
        )

        span_prob = {}
        span_prob['start'] = start_prob
        span_prob['end'] = end_prob

        return span_prob

    def eval_forward(self, video_feat, sent_feat, video_mask=None, sent_mask=None):
        B, T, D_v = video_feat.size()
        _, N, D_s = sent_feat.size()

        # Natural Language Modality
        word_feature, sent_embed = self.sentence_encoder(sent_feat)

        # Video Modality
        frame_feature = self.video_encoder(video_feat, word_feature)

        #Cross Modal
        cross_feat = self.CMI(frame_feature, word_feature, sent_embed)

        #Matching
        # match_prob, matching_feat = self.csmm(
        #     frame_feature, sent_embed, video_mask
        # )

        # Span Predictor
        # input_feat_to_span = match_prob.unsqueeze(dim=2) * cross_feat
        input_feat_to_span = cross_feat
        start_prob, end_prob = self.span_predictor(
            input_feat_to_span,
            v_mask=video_mask if self.video_if_mask else None,
        )

        span_prob = {}
        span_prob['start'] = start_prob
        span_prob['end'] = end_prob

        return span_prob
