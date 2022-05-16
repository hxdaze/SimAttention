import torch
import torch.nn as nn

from SimAttention.network.encoder import PCT_Encoder
from SimAttention.network.augmentation import Batch_PointWOLF
from SimAttention.utils.crops import *
from SimAttention.utils.provider import *
from SimAttention.dataloader import ModelNetDataSet
from torch.utils.data import DataLoader
from SimAttention.model import ProjectMLP, CrossedAttention
from SimAttention.model import SimAttention_4, SimAttention_5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root = '/home/akira/下载/Pointnet2_PyTorch-master/PCT/Point-Transformers-master/data/modelnet40_normal_resampled'
dataset = ModelNetDataSet(root)
trainDataLoader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

aug_method = Batch_PointWOLF()
online_encoder = PCT_Encoder().to(device)
crossed_method = CrossedAttention(1024).to(device)

model_5 = SimAttention_5(aug_method, b_fps, b_get_slice, b_get_cube, b_get_sphere,
                         online_encoder, crossed_method)

optimizer = torch.optim.SGD(model_5.parameters(), lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.3)

print('Start training...')
for epoch in range(0, 2):
    print('Epoch {} is running...'.format(epoch))
    model_5.train()
    for data, cls in trainDataLoader:
        points = data.numpy()
        points = random_point_dropout(points)
        points = random_scale_point_cloud(points)
        points = shift_point_cloud(points)
        points = torch.Tensor(points).to(device)
        optimizer.zero_grad()
        loss = model_5(points)
        print('loss is: ', loss)
        loss.backward()
        optimizer.step()
    scheduler.step()
