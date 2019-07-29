from torch.nn import CrossEntropyLoss
from loss.contrastive import ContrastiveLoss
from loss.triplet import TripletLoss
from loss.npair import NpairLoss
from loss.lifted import LiftedLoss
from loss.dmml import DMMLLoss


def make_loss(args, gids):
    """
    Construct loss function(s).
    """
    gid = None if gids is None else gids[0]
    if args.loss_type == 'softmax':
        criterion = CrossEntropyLoss()
    elif args.loss_type == 'contrastive':
        criterion = ContrastiveLoss(margin=args.margin)
    elif args.loss_type == 'triplet':
        criterion = TripletLoss(margin=args.margin)
    elif args.loss_type == 'softmax-triplet':
        criterion = {'softmax': CrossEntropyLoss(),
                     'triplet': TripletLoss(margin=args.margin)}
    elif args.loss_type == 'npair':
        criterion = NpairLoss(reg_lambda=0.002, gid=gid)
    elif args.loss_type == 'lifted':
        criterion = LiftedLoss(margin=args.margin, gid=gid)
    elif args.loss_type == 'dmml':
        criterion = DMMLLoss(num_support=args.num_support, distance_mode=args.distance_mode,
                             margin=args.margin, gid=gid)
    else:
        raise NotImplementedError

    return criterion
