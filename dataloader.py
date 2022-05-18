import os
import torch
import numpy as np
from torch.utils.data import Dataset
from SimAttention.utils.provider import pc_normalize

root = '/home/akira/下载/Pointnet2_PyTorch-master/byol_pcl/data/modelnet40_normal_resampled'


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


# data = ModelNetDataSet(root)
# DataLoader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True)
# flag = 3
# i = 0
#
# for point, cls in DataLoader:
#     print(point.shape, cls)
#     i += 1
#     if i == flag:
#         break
