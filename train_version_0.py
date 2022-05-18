import torch
import torch.nn as nn

from SimAttention.network.encoder import PCT_Encoder
from SimAttention.network.augmentation import Batch_PointWOLF
from SimAttention.utils.crops import *
from SimAttention.utils.provider import *
from SimAttention.dataloader import ModelNetDataSet
from torch.utils.data import DataLoader
from SimAttention.model import CrossedAttention
from SimAttention.model import SimAttention_5
import logging

# single GPU train
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)

# parameters
root = '/home/akira/下载/Pointnet2_PyTorch-master/PCT/Point-Transformers-master/data/modelnet40_normal_resampled'
EPOCH = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.01
aug_method = Batch_PointWOLF()
online_encoder = PCT_Encoder().to(device)
crossed_method = CrossedAttention(1024).to(device)

# load dataset
dataset = ModelNetDataSet(root, ,split='train')
trainDataLoader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
# load model
model_5 = SimAttention_5(aug_method, b_fps, b_get_slice, b_get_cube, b_get_sphere, online_encoder, crossed_method)

# check pretrained model
try:
    checkpoint = torch.load('trained_model.pth')
    start_epoch = checkpoint['epoch']
    model_5.load_state_dict(checkpoint['model_state_dict'])
    logger.info('Use pretrain model')
except:
    logger.info('No existing model, starting training from scratch...')
    start_epoch = 0

# optimizer
optimizer = torch.optim.SGD(model_5.parameters(), lr=LEARNING_RATE, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.3)

# train
logger.info('Start training...')
for epoch in range(start_epoch, EPOCH):
    logger.info('Epoch {} is running...'.format(epoch))
    model_5.train()
    mean_loss = 0.0
    for data, cls in trainDataLoader:
        points = data.numpy()
        points = random_point_dropout(points)
        points = random_scale_point_cloud(points)
        points = shift_point_cloud(points)
        points = torch.Tensor(points).to(device)
        optimizer.zero_grad()
        loss = model_5(points)
        mean_loss +=  loss.item()
        loss.backward()
        optimizer.step()
    scheduler.step()
    logger.info('Mean Loss for this epoch is: ', mean_loss / len(trainDataLoader))

 # when the train process is done, save the net
logger.info('Save Model...')
model_save_path = 'trained_model.pth'
logger.info('Saving at %s' % model_save_path)
state = {
    'epoch': EPOCH,
    'model_state_dict': model_5.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}
torch.save(state, model_save_path)

logger.info('End of training...')
