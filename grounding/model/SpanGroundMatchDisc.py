import torch
import torch.nn as nn
from torch.nn import functional as F

from .networks.attention import *
from .components import SentenceEncoder, VideoEncoder, SpanPredictor, CrossModalInteraction, TemporalOrderDiscriminator
from .components.DistributionAlign import VideoTextSemanticMatch

class GMD(nn.Module):
    def __init__(self, video_seq_set, sent_seq_set, grounding_set, matching_set, logger, drop_out):
        super(GMD, self).__init__()

        # Sentece Encoder
        sent_encoder = SentenceEncoder.select_sent_encoder(sent_seq_set['name'], logger)
        self.sentence_encoder = sent_encoder(
            sent_seq_set, logger
        )
        self.textual_dim = self.sentence_encoder.textual_dim

        # Video Encoder
        video_seq_set['query_dim'] = self.textual_dim
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

        # Cross-Modal Macthing
        matching_set['cross']['video_dim'] = self.visual_dim
        matching_set['cross']['query_dim'] = self.textual_dim
        self.csmm = VideoTextSemanticMatch(
            matching_set['cross'],
            matching_set['temporal'],
            matching_set['predict']
        )
        self.matching_dim = self.csmm.temporal_dim

        # Temporal Order Discriminator
        tod = TemporalOrderDiscriminator.select_temporal_order_discriminator(
            'moment_pooling',
            logger
        )
        self.tod = tod(self.visual_dim, logger)

    def forward(self, query_feat, query_mask,
                ori_video_feat, ori_video_mask,
                pseudo_video_feat, pseudo_video_mask,
                ori_temporal_mask, ori_fore_mask, ori_back_mask,
                pseudo_temporal_mask, pseudo_fore_mask, pseudo_back_mask):
        _, N, D_q = query_feat.size()

        # Natural Language Modality
        word_feat, sent_embed = self.sentence_encoder(query_feat)

        # Video Modality
        ori_frame_feat = self.video_encoder(ori_video_feat, word_feat)
        pseudo_frame_feat = self.video_encoder(pseudo_video_feat, word_feat)

        # Cross Modal
        ori_cross_feat = self.CMI(ori_frame_feat, word_feat, sent_embed)

        # Matching
        ori_match_prob, ori_matching_feat = self.csmm(
            ori_frame_feat, sent_embed, ori_video_mask
        )
        pseudo_match_prob, pseudo_matching_feat = self.csmm(
            pseudo_frame_feat, sent_embed, pseudo_video_mask
        )

        # Span Predictor
        ori_gated_feat = ori_match_prob.unsqueeze(dim=2) * ori_cross_feat
        start_prob, end_prob = self.span_predictor(
            ori_gated_feat,
            v_mask=ori_video_mask if self.video_if_mask else None,
        )
        span_prob = {}
        span_prob['start'] = start_prob
        span_prob['end'] = end_prob

        # Temporal Order Discriminator
        ori_disc_prob = self.tod(ori_frame_feat, ori_temporal_mask, ori_fore_mask, ori_back_mask)
        pseudo_disc_prob = self.tod(pseudo_frame_feat, pseudo_temporal_mask, pseudo_fore_mask, pseudo_back_mask)

        return span_prob, ori_match_prob, pseudo_match_prob, \
                ori_disc_prob, pseudo_disc_prob

    def eval_forward(self, video_feat, query_feat, video_mask=None, sent_mask=None):
        _, N, D_q = query_feat.size()

        # Natural Language Modality
        word_feat, sent_embed = self.sentence_encoder(query_feat)

        # Video Modality
        frame_feat = self.video_encoder(video_feat, word_feat)

        # Cross Modal
        cross_feat = self.CMI(frame_feat, word_feat, sent_embed)

        # Matching
        match_prob, matching_feat = self.csmm(
            frame_feat, sent_embed, video_mask
        )

        # Span Predictor
        gated_feat = match_prob.unsqueeze(dim=2) * cross_feat
        start_prob, end_prob = self.span_predictor(
            gated_feat,
            v_mask=video_mask if self.video_if_mask else None,
        )
        span_prob = {}
        span_prob['start'] = start_prob
        span_prob['end'] = end_prob

        return span_prob
