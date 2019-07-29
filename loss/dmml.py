import torch
import torch.nn as nn
from torch.nn import functional as F

import numbers

from loss.common import euclidean_dist, cosine_dist


class DMMLLoss(nn.Module):
    """
    DMML loss with center support distance and hard mining distance.

    Args:
        num_support: the number of support samples per class.
        distance_mode: 'center_support' or 'hard_mining'.
    """
    def __init__(self, num_support, distance_mode='hard_mining', margin=0.4, gid=None):
        super().__init__()

        if not distance_mode in ['center_support', 'hard_mining']:
            raise Exception('Invalid distance mode for DMML loss.')
        if not isinstance(margin, numbers.Real):
            raise Exception('Invalid margin parameter for DMML loss.')

        self.num_support = num_support
        self.distance_mode = distance_mode
        self.margin = margin
        self.gid = gid

    def forward(self, feature, label):
        feature = feature.cpu()
        label = label.cpu()
        classes = torch.unique(label)  # torch.unique() is cpu-only in pytorch 0.4
        if self.gid is not None:
            feature, label, classes = feature.cuda(self.gid), label.cuda(self.gid), classes.cuda(self.gid)
        num_classes = len(classes)
        num_query = label.eq(classes[0]).sum() - self.num_support

        support_inds_list = list(map(
            lambda c: label.eq(c).nonzero()[:self.num_support].squeeze(1), classes))
        query_inds = torch.stack(list(map(
            lambda c: label.eq(c).nonzero()[self.num_support:], classes))).view(-1)
        query_samples = feature[query_inds]

        if self.distance_mode == 'center_support':
            center_points = torch.stack([torch.mean(feature[support_inds], dim=0)
                for support_inds in support_inds_list])
            dists = euclidean_dist(query_samples, center_points)
        elif self.distance_mode == 'hard_mining':
            dists = []
            max_self_dists = []
            for i, support_inds in enumerate(support_inds_list):
                # dist_all = euclidean_dist(query_samples, feature[support_inds])
                dist_all = cosine_dist(query_samples, feature[support_inds])
                max_dist, _ = torch.max(dist_all[i*num_query:(i+1)*num_query], dim=1)
                min_dist, _ = torch.min(dist_all, dim=1)
                dists.append(min_dist)
                max_self_dists.append(max_dist)
            dists = torch.stack(dists).t()
            # dists = torch.clamp(torch.stack(dists).t() - self.margin, min=0.0)
            for i in range(num_classes):
                dists[i*num_query:(i+1)*num_query, i] = max_self_dists[i]

        log_prob = F.log_softmax(-dists, dim=1).view(num_classes, num_query, -1)

        target_inds = torch.arange(0, num_classes)
        if self.gid is not None:
            target_inds = target_inds.cuda(self.gid)
        target_inds = target_inds.view(num_classes, 1, 1).expand(num_classes, num_query, 1).long()

        dmml_loss = -log_prob.gather(2, target_inds).squeeze().view(-1).mean()

        batch_size = feature.size(0)
        l2_loss = torch.sum(feature ** 2) / batch_size
        dmml_loss += 0.002 * 0.25 * l2_loss

        return dmml_loss
