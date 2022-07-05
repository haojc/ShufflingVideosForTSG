import torch
import torch.nn as nn

def select_CMI(name, logger):
    if name.lower() in ['onlyvideo', 'a']:
        CMI = OnlyVideo
    elif name.lower() in ['videosentconcat', 'vs', 'b']:
        CMI = VideoSentenceConcat
    elif name.lower() in ['tall', 'mm', 'c']:
        CMI = TALL
    else:
        logger.error('error CMI name:', name,
                     'Must in a, b, c ')
    return CMI

class AbstartCMI(nn.Module):
    def __init__(self, video_dim, sent_dim, *args):
        super(AbstartCMI, self).__init__()
        self.video_dim = video_dim
        self.sent_dim = sent_dim

    def forward(self, video_feat, word_feat, sent_feat):
        return NotImplementedError

class OnlyVideo(AbstartCMI):
    def __init__(self, video_dim, sent_dim, *args):
        super(OnlyVideo, self).__init__(video_dim, sent_dim)
        self._cross_dim= video_dim

    def cross_dim(self):
        return self._cross_dim

    def forward(self, video_feat, word_feat, sent_feat):
        return video_feat

class VideoSentenceConcat(AbstartCMI):
    def __init__(self, video_dim, sent_dim, *args):
        super(VideoSentenceConcat, self).__init__(video_dim, sent_dim)
        self._cross_dim= video_dim + sent_dim

    def cross_dim(self):
        return self._cross_dim

    def forward(self, video_feat, word_feat, sent_feat):
        B, T, D_v = video_feat.size()
        cross_feat = torch.cat([video_feat, sent_feat.unsqueeze(1).expand(-1, T, -1)], dim=-1)
        return cross_feat

class TALL(AbstartCMI):
    def __init__(self, video_dim, sent_dim, *args):
        super(TALL, self).__init__(video_dim, sent_dim)
        assert video_dim == sent_dim
        self._cross_dim= self.crossmodal_dim = video_dim * 4

    def cross_dim(self):
        return self._cross_dim

    def forward(self, video_feat, word_feat, sent_feat):
        B, T, D_v = video_feat.size()
        _, D_s = sent_feat.size()
        assert D_s == D_v

        sent_feat = sent_feat.unsqueeze(1).expand(-1, T, -1)
        cross_feat = torch.cat(
            (video_feat, sent_feat, video_feat * sent_feat, video_feat + sent_feat),
            -1)
        return cross_feat