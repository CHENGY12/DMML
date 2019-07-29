import torch


def euclidean_dist(x, y, squared=True):
    """
    Compute (Squared) Euclidean distance between two tensors.

    Args:
        x: input tensor with size N x D.
        y: input tensor with size M x D.

        return: distance matrix with size N x M.
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception('Invalid input shape.')

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    dist = torch.pow(x - y, 2).sum(2)

    if squared:
        return dist
    else:
        return torch.sqrt(dist+1e-12)


def cosine_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception('Invalid input shape.')

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    # dist = -torch.mul(x, y).sum(2) / torch.clamp(torch.mul(torch.norm(x, p=2, dim=2), torch.norm(y, p=2, dim=2)), min=1e-6)
    dist = -torch.mul(x, y).sum(2)

    return dist


def get_mask(label, mask_type='positive'):
    """
    Generate positive and negative masks for contrastive and triplet loss.
    """
    device = label.device
    identity = torch.eye(label.shape[0]).byte()
    not_identity = ~identity
    not_identity = not_identity.to(device)

    if mask_type == 'positive':
        mask = torch.eq(label.unsqueeze(1), label.unsqueeze(0))
    elif mask_type == 'negative':
        mask = torch.ne(label.unsqueeze(1), label.unsqueeze(0))
    mask = mask.byte()
    mask = mask & not_identity

    return mask
