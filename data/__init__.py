from torchvision import transforms
from torch.utils.data import dataloader

import numpy as np
import random

from data.market1501 import Market1501
from data.duke import DukeMTMC_reID
from data.sampler import RandomSampler
from data.random_erasing import RandomErasing


def make_dataloader(args, epoch=0):
    """
    Make train dataloader.
    
    Args:
        epoch: current epoch number, used in random erasing data augmentation.
    """
    train_list = [
        transforms.Resize((args.img_height, args.img_width), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    if args.random_erasing:
        probability = 0.3 + 0.4*min((float(epoch)/args.num_epochs), 0.8)
        s_epoch = 0.1 + 0.3*min((float(epoch)/args.num_epochs), 0.8)
        train_list.append(RandomErasing(probability=probability, s_epoch=s_epoch, mean=[0.0, 0.0, 0.0]))
    train_transform = transforms.Compose(train_list)

    batch_m = args.num_classes
    if 'dmml' in args.loss_type:
        batch_k = args.num_support + args.num_query
    elif args.loss_type == 'npair':
        batch_k = 2
    else:
        batch_k = args.num_instances

    if args.dataset == 'market1501':
        train_set = Market1501(args.dataset_root, train_transform, split='train')
    elif args.dataset == 'duke':
        train_set = DukeMTMC_reID(args.dataset_root, train_transform, split='train')
    else:
        raise NotImplementedError

    train_loader = dataloader.DataLoader(train_set,
                       sampler=RandomSampler(train_set, batch_k),
                       batch_size=batch_m*batch_k,
                       num_workers=args.num_workers,
                       drop_last=True)

    return train_loader
