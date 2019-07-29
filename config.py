import argparse

from utils import float_or_string


def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset',
                        type=str,
                        help='dataset name, which can be chosen from \'market1501\' and \'duke\'',
                        default='market1501')

    parser.add_argument('--dataset_root',
                        type=str,
                        help='dataset path',
                        default='../dataset')

    parser.add_argument('--exp_root',
                        type=str,
                        help='path to store models and logs',
                        default='./exp')

    parser.add_argument('--num_epochs',
                        type=int,
                        help='number of training epochs',
                        default=1200)

    parser.add_argument('--lr',
                        type=float,
                        help='learning rate',
                        default=2e-4)

    parser.add_argument('--lr_decay_start_epoch',
                        type=int,
                        help='epoch from when learning rate starts to decay exponentially',
                        default=600)

    parser.add_argument('--weight_decay',
                        type=float,
                        help='L2 weight decay',
                        default=1e-4)

    parser.add_argument('--num_classes',
                        type=int,
                        help='number of classes per episode',
                        default=32)

    parser.add_argument('--num_support',
                        type=int,
                        help='number of support samples per class (for DMML)',
                        default=5)

    parser.add_argument('--num_query',
                        type=int,
                        help='number of query samples per class (for DMML)',
                        default=1)

    parser.add_argument('--num_instances',
                        type=int, 
                        help='number of instances per class (not for DMML)',
                        default=6)

    parser.add_argument('--distance_mode',
                        type=str,
                        help='distance measurement method of DMML, \
                             which can be chosen from \'center_support\' and \'hard_mining\'',
                        default='hard_mining')

    parser.add_argument('--margin',
                        type=float_or_string,
                        help='margin parameter for contrastive loss, triplet loss or DMML loss',
                        default=.0)

    parser.add_argument('--img_height',
                        type=int,
                        help='height of resized input images',
                        default=256)

    parser.add_argument('--img_width',
                        type=int,
                        help='width of resized input images',
                        default=128)

    parser.add_argument('--loss_type',
                        type=str,
                        help='loss used in training, can be chosen from \'softmax\', \'contrastive\', \
                        \'triplet\', \'softmax-triplet\', \'lifted\',  \'npair\' and \'dmml\'',
                        default='dmml')

    parser.add_argument('--alpha',
                        type=float,
                        help='balance parameter of softmax loss and triplet loss, \
                        ranging from 0 to 1.0',
                        default=0.5)

    parser.add_argument('--remove_downsample',
                        action='store_true',
                        help='whether to remove the final downsample operation in resnet50 model')

    parser.add_argument('--random_erasing',
                        action='store_true',
                        help='whether to use random erasing for data augmentation')

    parser.add_argument('--num_workers',
                        type=int,
                        help='number of subprocesses for data loading',
                        default=0)

    parser.add_argument('--manual_seed',
                        type=int,
                        help='manual seed for initialization',
                        default=7)

    parser.add_argument('--cuda',
                        action='store_true',
                        help='whether to use cuda')

    parser.add_argument('--gpu',
                        type=str,
                        help='id of gpu device(s) to be used',
                        default='0, 1')

    return parser
