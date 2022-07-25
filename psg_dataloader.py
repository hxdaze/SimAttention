import os
import json
import numpy as np
import torch.utils.data
from torch.utils.data import Dataset
from network.augmentation import PointWOLF
from utils.provider import pc_normalize
from utils.visualize import visualization


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = np.eye(num_classes, dtype=np.float32)[y, ] # B, 1, 16
    return new_y


def translate_point_cloud(point_cloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_point_cloud = np.add(np.multiply(point_cloud, xyz1), xyz2).astype('float32')
    return translated_point_cloud


def random_dropout(pc, max_dropout_ratio=0.875):
    dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
    drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
    if len(drop_idx) > 0:
        pc[drop_idx, :] = pc[0, :]
    return pc


class PartNormalDataset(Dataset):
    def __init__(self, root,
                 npoints=2500,
                 split='train',
                 class_choice=None,
                 normal_channel=False):

        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.aug1 = PointWOLF(0.4)
        self.aug2 = PointWOLF(0.8)
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:
            aug1, aug2, cls_1_hot, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            cls_1_hot = to_categorical(cls, 16)
            data = np.loadtxt(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
                # take transforms in dataloader:
            point_set = random_dropout(point_set)
            point_set = translate_point_cloud(point_set)
            # add augmentation
            _, aug1 = self.aug1(point_set)
            _, aug2 = self.aug2(point_set)
            # seg
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (aug1, aug2, cls_1_hot, seg)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        aug1 = aug1[choice, :]
        aug2 = aug2[choice, :]
        seg = seg[choice]

        # aug B, 2500, 3
        # cls B, 1, 16
        # seg B, 2500
        return aug1, aug2, cls_1_hot, seg

    def __len__(self):
        return len(self.datapath)


if __name__ == "__main__":
    file_path = r'/home/haruki/下载/shapenet/shapenetcore'
    dataset = PartNormalDataset(file_path)
    Dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    demo = 0
    l = 0
    for morph_1, morph_2, class_label, seg_label in Dataloader:
        demo = morph_2[2].numpy()
        l = class_label
        print(morph_1.shape, morph_2.shape, class_label.shape, seg_label.shape)
        print(l)
        break
    # seg problem
    visualization(demo)
