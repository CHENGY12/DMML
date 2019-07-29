import torch
from torchvision import transforms

import os
import random
import numpy as np
from collections import defaultdict

from data.market1501 import Market1501
from data.duke import DukeMTMC_reID
from model import resnet_model

from utils import get_id, flip_img, compute_map


def eval(gid, dataset, dataset_root, which, exp_dir):
    mAP, CMC = main(gid=gid, dataset=dataset, dataset_root=dataset_root, which=which, exp_dir=exp_dir, verbose=False)
    return mAP, CMC


def main(gid=None, dataset=None, dataset_root=None, which=None, exp_dir=None, verbose=False):
    """
    Configs
    """
    GPU_ID = 0                         # gpu id or 'None'
    BATCH_SIZE = 32                    # batch size when extracting query and gallery features
    IMG_SIZE = (256, 128)
    DATASET = 'market1501'             # market1501, duke
    WHICH = 'last'                     # which model to load
    EXP_DIR = './exp/dmml/market1501'
    NORMALIZE_FEATURE = True           # whether to normalize features in evaluation
    NUM_WORKERS = 8

    if gid is not None:
        GPU_ID = gid
    if dataset is not None:
        DATASET = dataset
    if which is not None:
        WHICH = which
    if exp_dir is not None:
        EXP_DIR = exp_dir

    """
    Datasets
    """
    if dataset_root is None:
        # change dataset directories here to your own if needed
        if DATASET == 'market1501':
            dataset_root = '<DATASET_ROOT_MARKET>'
        elif DATASET == 'duke':
            dataset_root = '<DATASET_ROOT_DUKE>'
        else:
            raise NotImplementedError

    print('Generating dataset...')
    eval_transform = transforms.Compose([transforms.Resize(IMG_SIZE, interpolation=3),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    if DATASET == 'market1501':
        datasets = {x: Market1501(dataset_root, transform=eval_transform, split=x)
                    for x in ['gallery', 'query']}
        num_classes = 751
    elif DATASET == 'duke':
        datasets = {x: DukeMTMC_reID(dataset_root, transform=eval_transform, split=x)
                    for x in ['gallery', 'query']}
        num_classes = 702

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=BATCH_SIZE,
               shuffle=False, num_workers=NUM_WORKERS) for x in ['gallery', 'query']}
    print('Done.')

    """
    Model
    """
    print('Restoring model...')

    ### You may need to modify the arguments of the model according to your training settings.

    model = resnet_model(remove_downsample=True)
    # model = resnet_model(num_classes=num_classes, include_top=False, remove_downsample=False)
    
    model.load_state_dict(torch.load('{}/model_{}.pth'.format(EXP_DIR, WHICH)))
    if GPU_ID is not None:
        model.cuda(GPU_ID)
    model.eval()
    print('Done.')

    """
    Test
    """
    print('Getting image ID...')
    gallery_cam, gallery_label = get_id(datasets['gallery'].imgs, dataset=DATASET)
    query_cam, query_label = get_id(datasets['query'].imgs, dataset=DATASET)
    print('Done.')

    # Extract feature
    print('Extracting gallery feature...')
    gallery_feature, g_images = extract_feature(model, dataloaders['gallery'],
        normalize_feature=NORMALIZE_FEATURE, gid=GPU_ID, verbose=verbose)
    print('Done.')
    print('Extracting query feature...')
    query_feature, q_images = extract_feature(model, dataloaders['query'],
        normalize_feature=NORMALIZE_FEATURE, gid=GPU_ID, verbose=verbose)
    print('Done.')

    query_cam = np.array(query_cam)
    query_label = np.array(query_label)
    gallery_cam = np.array(gallery_cam)
    gallery_label = np.array(gallery_label)

    # Evaluate
    print('Evaluating...')
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], query_cam[i],
            gallery_feature, gallery_label, gallery_cam)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC / len(query_label)  # average CMC
    print('Done.')
    print('Rank-1: {:.6f} Rank-5: {:.6f} Rank-10: {:.6f} mAP: {:.6f}'.format(
        CMC[0].item(), CMC[4].item(), CMC[9].item(), ap/len(query_label)))

    return ap / len(query_label), CMC


def extract_feature(model, dataloaders, normalize_feature=True, gid=None, verbose=False):
    """
    Extract query and gallery features.
    """
    features = torch.FloatTensor()
    count = 0

    images_numpy = None

    for (image, label) in dataloaders:
        n, c, h, w = image.size()
        count += n
        if count % (10 * n) == 0 and verbose:
            print(count)
        ff = torch.FloatTensor(n, 2048).zero_()
        for i in range(2):
            if i == 1:
                image = flip_img(image.cpu())
            if gid is not None:
                image = image.cuda(gid)
            feat = model(image) 
            f = feat.data.cpu()
            ff = ff + f
        # normalize feature
        if normalize_feature:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff), 0)

        img_numpy = image.cpu().numpy()
        if images_numpy is not None:
            images_numpy = np.append(images_numpy, img_numpy, axis=0)
        else:
            images_numpy = img_numpy

    print('total: {:d}'.format(count))

    return features, images_numpy


def evaluate(qf, ql, qc, gf, gl, gc):
    """
    Evaluation
    """
    query = qf.view(-1, 1)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()

    # prediction index
    index = np.argsort(score)
    index = index[::-1]

    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
    ap_tmp, CMC_tmp = compute_map(index, good_index, junk_index)
    return ap_tmp, CMC_tmp


if __name__ == '__main__':
    main()
