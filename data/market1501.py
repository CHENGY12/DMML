import torch.utils.data as Data
from torchvision.datasets.folder import default_loader

from os.path import join as ospj

from data.common import list_pictures


class Market1501(Data.Dataset):
    """
    Market-1501 dataset for person re-identification.
    """
    def __init__(self, dataset_root, transform, split='train'):
        if not split in ['train', 'gallery', 'query']:
            raise Exception('Invalid dataset split.')
        self.transform = transform
        self.loader = default_loader
        self.split = split

        if split == 'train':
            data_path = ospj(dataset_root, 'bounding_box_train')
        elif split == 'gallery':
            data_path = ospj(dataset_root, 'bounding_box_test')
        elif split == 'query':
            data_path = ospj(dataset_root, 'query')
        
        self.imgs = [path for path in list_pictures(data_path)]

        if split == 'train':
            self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}

    def __getitem__(self, index):
        path = self.imgs[index]
        label = self._id2label[self.id(path)] if self.split == 'train' else self.id(path)

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def id(file_path):
        """
        file_path: unix style file path
        return: person id
        """
        return int(file_path.split('/')[-1].split('_')[0])

    @staticmethod
    def camera(file_path):
        """
        file_path: unix style file path
        return: camera id
        """
        return int(file_path.split('/')[-1].split('_')[1][1])

    @property
    def ids(self):
        """
        return: person id list corresponding to dataset image paths
        """
        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):
        """
        return: unique person ids in ascending order
        """
        return sorted(set(self.ids))

    @property
    def cameras(self):
        """
        return: camera id list corresponding to dataset image paths
        """
        return [self.camera(path) for path in self.imgs]
