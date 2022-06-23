from dataloader import LatentRepresentationDataSet
from network.shape_classifier_2 import ShapeClassifier_2
import torch
import sys
from tqdm import tqdm
import math
import numpy as np
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

learning_rate = 0.001
max_epochs = 200
root = r'/home/akira/下载/Pointnet2_PyTorch-master/SimAttention/jupyter_tests/mydata'

lrds = LatentRepresentationDataSet(root)
train_data_loader = torch.utils.data.DataLoader(lrds, batch_size=1, shuffle=True)

classifier = ShapeClassifier_2().to(device)
criterion = torch.nn.CrossEntropyLoss()
trainDataLoader = tqdm(train_data_loader, file=sys.stdout)
optimizer = torch.optim.SGD(classifier.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.005)
lf = lambda x: ((1 + math.cos(x * math.pi / max_epochs)) / 2) * (1 - 0.01) + 0.01  # cosine
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
tb_writer = SummaryWriter()

mean_correct = []
tags = ["train_acc", "learning_rate"]

for epoch in range(0, max_epochs):
    for f, l in trainDataLoader:
        f, l = f.to(device), l.to(device)
        f = f.reshape(-1, 1024)
        l = l.reshape(l.shape[-1])
        # print('l shape: ', l.shape) # torch.Size([8])
        # print('f shape: ', f.shape) # torch.Size([8, 1024])

        # optimizer.zero_grad()
        pred = classifier(f.float())
        # print('pred shape: ', pred.shape) # torch.Size([8, 40])
        loss = criterion(pred, l.long())

        pred_choice = pred.data.max(1)[1]
        # print('pred_choice shape: ', pred_choice.shape)
        correct = pred_choice.eq(l.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(f.size()[0]))
        # print("loss is: ", loss.item())
        loss.backward()
        optimizer.step()
    scheduler.step()
    train_instance_acc = np.mean(mean_correct)
    # print('Train Instance Accuracy: %f' % train_instance_acc)
    tb_writer.add_scalar(tags[0], train_instance_acc, epoch)
    tb_writer.add_scalar(tags[1], optimizer.param_groups[0]["lr"], epoch)

