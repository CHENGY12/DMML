import torch
import torch.nn as nn

from loss.common import euclidean_dist


class LiftedLoss(nn.Module):
    """
    Lifted loss.
    """
    def __init__(self, margin=0.4, gid=None):
        super(LiftedLoss, self).__init__()
        self.margin = margin
        self.gid = gid

    def forward(self, features, labels):
        batch_size = labels.size(0)
        positive_mask = labels.view(1, -1) == labels.view(-1, 1)
        negative_mask = ~positive_mask
        dists = euclidean_dist(features, features, squared=False)

        dists_repeated_x = dists.repeat(1,batch_size).view(-1,batch_size)
        negative_mask_repeated_x = negative_mask.repeat(1,batch_size).view(-1,batch_size).float()
        dists_repeated_y = dists.transpose(1,0).repeat(batch_size,1)
        negative_mask_repeated_y = negative_mask.transpose(1,0).repeat(batch_size,1).float()
        positive_dists = dists.view(-1, 1)
        J_matrix = torch.log(torch.sum(torch.exp(self.margin - dists_repeated_x) * negative_mask_repeated_x,1,keepdim=True) + torch.sum(torch.exp(self.margin - dists_repeated_y) * negative_mask_repeated_y,1,keepdim=True)) + positive_dists
        J_matrix_valid = torch.masked_select(J_matrix, positive_mask.view(-1, 1))
        if self.gid is not None:
            J_matrix_valid = torch.max(torch.zeros(J_matrix_valid.size()).cuda(self.gid), J_matrix_valid)
        else:
            J_matrix_valid = torch.max(torch.zeros(J_matrix_valid.size()), J_matrix_valid)
        lifted_loss_matrix = J_matrix_valid * J_matrix_valid
        lifted_loss = torch.sum(lifted_loss_matrix) / (2 * positive_mask.sum().item())

        return lifted_loss
