import torch
import torch.nn as nn
from torch.nn import functional as F


class NpairLoss(nn.Module):
    """
    Multi-class N-pair loss.

    Args:
        reg_lambda: L2 norm regularization for embedding vectors.
    """
    def __init__(self, reg_lambda=0.002, gid=None):
        super(NpairLoss, self).__init__()
        self.reg_lambda = reg_lambda
        self.gid = gid

    def forward(self, feature, label):
        feature = feature.cpu()
        label = label.cpu()
        classes = torch.unique(label)  # torch.unique() is cpu-only in pytorch 0.4
        if self.gid is not None:
            feature, label, classes = feature.cuda(self.gid), label.cuda(self.gid), classes.cuda(self.gid)

        anchor_inds = torch.stack(list(map(
            lambda c: label.eq(c).nonzero()[0].squeeze(0), classes)))
        positive_inds = torch.stack(list(map(
            lambda c: label.eq(c).nonzero()[1].squeeze(0), classes)))
        anchor = feature[anchor_inds]
        positive = feature[positive_inds]

        batch_size = anchor.size(0)
        classes = classes.view(classes.size(0), 1)
        target = (classes == classes.t()).float()
        target = target / torch.sum(target, dim=1, keepdim=True).float()

        similarity = torch.matmul(anchor, positive.t())
        ce_loss = torch.mean(torch.sum(-target * F.log_softmax(similarity, -1), -1))
        l2_loss = torch.sum(anchor**2) / batch_size + torch.sum(positive**2) / batch_size

        npair_loss = ce_loss + l2_loss * self.reg_lambda * 0.25
        return npair_loss
