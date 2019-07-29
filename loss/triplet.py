import torch
import torch.nn as nn
from torch.nn import functional as F

import numbers

from loss.common import euclidean_dist
from loss.common import get_mask


class TripletLoss(nn.Module):
    """
    Batch hard triplet loss.
    """
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        if not (isinstance(margin, numbers.Real) or margin == 'soft'):
            raise Exception('Invalid margin parameter for triplet loss.')
        self.margin = margin

    def forward(self, feature, label):
        distance = euclidean_dist(feature, feature, squared=False)

        positive_mask = get_mask(label, 'positive')
        hardest_positive = (distance * positive_mask.float()).max(dim=1)[0]
        
        negative_mask = get_mask(label, 'negative')
        max_distance = distance.max(dim=1)[0]
        not_negative_mask = ~(negative_mask.data)
        negative_distance = distance + max_distance * (not_negative_mask.float())
        hardest_negative = negative_distance.min(dim=1)[0]

        diff = hardest_positive - hardest_negative
        if isinstance(self.margin, numbers.Real):
            tri_loss = (self.margin + diff).clamp(min=0).mean()
        else:
            tri_loss = F.softplus(diff).mean()
        
        return tri_loss
