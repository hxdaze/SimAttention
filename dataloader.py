import os
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.provider import pc_normalize
from utils.provider import feature_norm


class ModelNetDataSet(Dataset):
    def __init__(self, root, split='train', cache_size=15000):
        super().__init__()
        self.root = root
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {'train': [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))],
                     'test': [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]}

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            point_set = point_set[:, 0:3]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)


class LatentRepresentationDataSet(Dataset):
    def __init__(self, root, cache_size=15000):
        super().__init__()
        self.root = root
        # get the number of the file
        # todo: 如果不对，下面一行就不用剪去1
        self.number_txt = len(os.listdir(self.root)) - 1
        # get all the data path of txt
        self.datapath = [os.path.join(self.root, str(i) + '.txt') for i in range(self.number_txt)]
        print('The size of %s data is %d' % (split, self.number_txt))

        self.cache_size = cache_size  
        self.cache = {} 

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            features, labels = self.cache[index]
        else:
            fn = self.datapath[index]
            data = np.genfromtxt(fn, delimiter=',').astype(np.float64)
            features = data[:, :-1]
            features = feature_norm(features)
            features = features.squeeze()
            labels = data[:, -1].astype(int)

            if len(self.cache) < self.cache_size:
                self.cache[index] = (features, labels)

        return features, labels

    def __getitem__(self, index):
        return self._get_item(index)
    
