import torch
import torch.nn as nn

import numbers

from loss.common import euclidean_dist
from loss.common import get_mask


class ContrastiveLoss(nn.Module):
    """
    Batch hard contrastive loss.
    """
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        if not isinstance(margin, numbers.Real):
            raise Exception('Invalid margin parameter for contrastive loss.')
        self.margin = margin

    def forward(self, feature, label):
        distance = euclidean_dist(feature, feature, squared=False)

        positive_mask = get_mask(label, 'positive')
        hardest_positive = (distance * positive_mask.float()).max(dim=1)[0]
        p_loss = hardest_positive.mean()
        
        negative_mask = get_mask(label, 'negative')
        max_distance = distance.max(dim=1)[0]
        not_negative_mask = ~negative_mask
        negative_distance = distance + max_distance * (not_negative_mask.float())
        hardest_negative = negative_distance.min(dim=1)[0]
        n_loss = (self.margin - hardest_negative).clamp(min=0).mean()

        con_loss = p_loss + n_loss
        return con_loss
