import os
import torch
import numpy as np

root = '/home/akira/下载/Pointnet2_PyTorch-master/byol_pcl/data/modelnet40_normal_resampled'


class ModelNetDataSet():
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.split = 'train'
        self.shape_ids = {'train': [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))],
                          'test': [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]}

        self.shape_names = ['_'.join(x.split('_')[0:-1]) for x in self.shape_ids[self.split]]
        self.datapath = [(self.shape_names[i], os.path.join(root, self.shape_names[i], self.shape_ids[self.split][i]) + '.txt')
                     for i in range(len(self.shape_ids[self.split]))]

        print(f'{len(self.datapath)} point clouds found.')

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        path = self.datapath[index][1]
        point_set = np.loadtxt(path, delimiter=',').astype(np.float32)[:, 0:3]
        return point_set


# data = ModelNetDataSet(root)
# DataLoader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True)
# flag = 3
# i = 0
#
# for point in DataLoader:
#     print(point.shape)
#     i += 1
#     if i == flag:
#         break
