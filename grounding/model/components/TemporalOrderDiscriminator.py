import torch
import torch.nn as nn
from ..networks.attention import mask_logits

def select_temporal_order_discriminator(name, logger):
    name = name.lower()
    if name in ['moment_pooling', 'mp']:
        tod = MomentPooling
    else:
        logger.error('error video encoder name:', name,
                     'Must in \'moment_pooling\', ')
    return tod


class MomentPooling(nn.Module):
    def __init__(self, visual_dim, logger, *args):
        super(MomentPooling, self).__init__()

        self.foreback_context = nn.Sequential(
            nn.Linear(visual_dim * 2, visual_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(p=0.5)

        self.fc_classifier_domain_video = nn.Sequential(
            nn.Linear(visual_dim * 3, 2),
        )

    def average_mask(self, feat, mask):
        return torch.sum(mask_logits(feat, mask, mask_value=0.0), dim=1) \
            / (torch.sum(mask, dim=1, keepdim=True) + 1e-6)

    def forward(self, feat, target_mask, fore_mask, back_mask):
        target_moment_feat = self.average_mask(feat, target_mask)
        fore_context_feat = self.average_mask(feat, fore_mask)
        back_context_feat = self.average_mask(feat, back_mask)

        fore_feat = self.foreback_context(torch.cat((fore_context_feat, target_moment_feat), -1))
        back_feat = self.foreback_context(torch.cat((target_moment_feat, back_context_feat), -1))

        concat_feat = torch.cat((target_moment_feat, fore_feat, back_feat), -1)
        video_prob = self.fc_classifier_domain_video(
            self.dropout(concat_feat)
        )

        return video_prob