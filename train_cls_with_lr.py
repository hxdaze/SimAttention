import sys
import math
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataloader import LatentRepresentationDataSet
from network.shape_classifier_2 import ShapeClassifier_2


learning_rate = 0.001
max_epochs = 200
# 存储latent representation的文件路径
root = r'/home/akira/下载/Pointnet2_PyTorch-master/SimAttention/jupyter_tests/mydata'
lrds = LatentRepresentationDataSet(root)
# 这里的batch_size就设置为1就好，因为读进来的数据其实就是bs=16的了，所以不需要额外设置！
train_data_loader = torch.utils.data.DataLoader(lrds, batch_size=1, shuffle=True)
trainDataLoader = tqdm(train_data_loader, file=sys.stdout)
tb_writer = SummaryWriter()

# build classification network strcuture
classifier = ShapeClassifier_2().cuda()
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(classifier.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.005)
lf = lambda x: ((1 + math.cos(x * math.pi / max_epochs)) / 2) * (1 - 0.01) + 0.01  # cosine
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

mean_correct = []
tags = ["train_acc", "train_loss", "learning_rate"]

for epoch in range(0, max_epochs):
    for f, l in trainDataLoader:
        f, l = f.cuda(), l.cuda()
        f = f.reshape(-1, 1024)
        l = l.reshape(l.shape[-1])
        optimizer.zero_grad()
        pred = classifier(f.float())
        loss = criterion(pred, l.long())
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(l.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(f.size()[0]))
        loss.backward()
        optimizer.step()
    scheduler.step()
    train_instance_acc = np.mean(mean_correct)
    tb_writer.add_scalar(tags[0], train_instance_acc, epoch)
    tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

