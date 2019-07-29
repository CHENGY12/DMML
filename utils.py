import torch

import numpy as np


def float_or_string(arg):
    """
    Tries to convert the string to float, otherwise returns the string.
    """
    try:
        return float(arg)
    except (ValueError, TypeError):
        return arg


def flip_img(img):
    """
    Do horizontal flipping for a given image.
    """
    inv_idx = torch.arange(img.size(3)-1, -1, -1).long()  # N x C x H x W
    img_flipped = img.index_select(3, inv_idx)
    return img_flipped


def get_id(img_path, dataset):
    """
    Get camera ids and class labels of images.
    """
    camera_id = []
    labels = []
    for path in img_path:
        filename = path.split('/')[-1]
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels


def compute_map(index, good_index, junk_index):
    """
    Compute mean average precision (mAP) and cumulative matching characteristic (CMC) scores.
    """
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i+1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc
