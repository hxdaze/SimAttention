import os
import torch
import numpy as np
from utils.provider import pc_normalize
from torch.utils.data import Dataset
from network.augmentation import PointWOLF


def random_dropout(pc, max_dropout_ratio=0.875):
    dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
    drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
    if len(drop_idx) > 0:
        pc[drop_idx, :] = pc[0, :]
    return pc


def translate_point_cloud(point_cloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_point_cloud = np.add(np.multiply(point_cloud, xyz1), xyz2).astype('float32')
    return translated_point_cloud


class AugModelNetDataSet(Dataset):
    def __init__(self, root, split='train', cache_size=15000):
        super().__init__()
        self.root = root
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        # augmentation function
        self.aug = PointWOLF(0.7)

        shape_ids = {'train': [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))],
                     'test': [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]}

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))
        self.cache_size = cache_size
        self.cache = {}

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            aug1, aug2, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.genfromtxt(fn[1], delimiter=',').astype(np.float32)
            point_set = point_set[:, 0:3]
            # take transforms in dataloader:
            point_set = random_dropout(point_set)
            point_set = translate_point_cloud(point_set)
            # add augmentation
            _, aug1 = self.aug(point_set)
            _, aug2 = self.aug(point_set)

            aug1 = pc_normalize(aug1)
            aug2 = pc_normalize(aug2)

            if len(self.cache) < self.cache_size:
                self.cache[index] = (aug1, aug2, cls)
        return aug1, aug2, cls

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == "__main__":
    root = '/home/haruki/下载/modelnet40_normal_resampled'
    aug_dataset = AugModelNetDataSet(root, 'train')
    trainDataLoader = torch.utils.data.DataLoader(aug_dataset, batch_size=4,
                                              shuffle=True, num_workers=4,
                                              pin_memory=True)
    break_flag = 3
    i = 0
    for aug1, aug2, cls in trainDataLoader:
        print(aug1.shape)
        i += 1
        if i == break_flag:
            break
    print("Test for new dataloader!")
