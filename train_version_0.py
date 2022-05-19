from SimAttention.network.encoder import PCT_Encoder
from SimAttention.network.augmentation import Batch_PointWOLF
from SimAttention.utils.crops import *
from SimAttention.utils.provider import *
from SimAttention.dataloader import ModelNetDataSet
from torch.utils.data import DataLoader
from SimAttention.model import CrossedAttention
from SimAttention.model import SimAttention_5
import logging
import torch
from tqdm import tqdm
import sys

logger = logging.getLogger(__name__)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
LEARNING_RATE = 0.01
EPOCH = 2

root = '/home/akira/下载/Pointnet2_PyTorch-master/PCT/Point-Transformers-master/data/modelnet40_normal_resampled'
train_dataset = ModelNetDataSet(root, split='train')
# test_dataset = ModelNetDataSet(root, split='test')
trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

aug_method = Batch_PointWOLF()
online_encoder = PCT_Encoder().to(device)
crossed_method = CrossedAttention(1024).to(device)

model_5 = SimAttention_5(aug_method, b_fps, b_get_slice,
                         b_get_cube, b_get_sphere,
                         online_encoder, crossed_method)

try:
    checkpoint = torch.load('trained_model.pth')
    start_epoch = checkpoint['epoch']
    model_5.load_state_dict(checkpoint['model_state_dict'])
    logger.info('Use pretrain model')
except:
    logger.info('No existing model, starting training from scratch...')
    start_epoch = 0

optimizer = torch.optim.SGD(model_5.parameters(), lr=LEARNING_RATE, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.3)

logger.info('Start training...')
for epoch in range(start_epoch, EPOCH):
    logger.info('Epoch {} is running...'.format(epoch))
    trainDataLoader = tqdm(trainDataLoader, file=sys.stdout)
    ten_step_loss = 0.0
    for step, data in enumerate(trainDataLoader):
        points, _ = data
        points = points.numpy()
        points = random_point_dropout(points)
        points = random_scale_point_cloud(points)
        points = shift_point_cloud(points)
        points = torch.Tensor(points).to(device)
        optimizer.zero_grad()
        loss = model_5(points)

        if step > 0 and step % 10 == 0:
            ten_step_loss += loss.item()
            print('==========10 steps mean loss==========')
            print("{} step mean loss is: ".format(step), ten_step_loss / 10)
            ten_step_loss = 0.0
        else:
            ten_step_loss += loss.item()
            print("{} step loss is: ".format(step), loss.item())

        loss.backward()
        optimizer.step()
    scheduler.step()

# when the train process is done, save the net
logger.info('Save Model...')
model_save_path = 'trained_model.pth'
logger.info('Saving at %s' % model_save_path)
state = {
    'epoch': EPOCH,
    'model_state_dict': model_5.module.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}
torch.save(state, model_save_path)

logger.info('End of training...')
