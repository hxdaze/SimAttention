import torch
import argparse
import math
from psg_model import SimSeg
from psg_dataloader import PartNormalDataset


def get_optimizer_and_scheduler(model, args):
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.005)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    return optimizer, scheduler


def get_dataloader(args):
    train_data_set = PartNormalDataset(args.root, split='trainval')
    train_dataLoader = torch.utils.data.DataLoader(train_data_set,
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  num_workers=4,
                                                  pin_memory=True)
    return train_dataLoader


def train_one_epoch(model, optimizer, data_loader):
    model.train()
    optimizer.zero_grad()

    for step, data in enumerate(data_loader):
        aug1, aug2, cls, _ = data
        aug1 = aug1.to(device)
        aug2 = aug2.to(device)
        cls = cls.to(device)
        # print('aug device: ', aug2.device)
        loss = model(aug1, aug2, cls)
        loss = loss.mean()
        print("step {} loss is: ".format(step), loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=r'/home/haruki/下载/shapenet/shapenetcore')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.001)
    opt = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SimSeg().to(device)
    optimizer, scheduler = get_optimizer_and_scheduler(model, opt)
    dataloader = get_dataloader(opt)
    print(opt)

    for epoch in range(opt.epochs):
        train_one_epoch(model, optimizer, dataloader)
