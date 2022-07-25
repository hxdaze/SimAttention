import os
import math
import sys
import torch
import argparse
import tempfile
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.train_eval_utils import is_main_process, evaluate, reduce_value
from utils.distributed_utils import dist, cleanup
from psg_model import *
from psg_dataloader import PartNormalDataset


def init_process(rank, world_size, args):
    # 初始化各进程环境 start
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    args.rank = rank
    args.world_size = world_size
    args.gpu = rank
    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()

    rank = args.rank
    args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增
    checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
    device = torch.device(args.device)
    return checkpoint_path, rank, device


def get_sampler_and_dataloader(rank, args):
    # 实例化训练数据集
    train_data_set = PartNormalDataset(args.root, split='trainval')
    test_data_set = PartNormalDataset(args.root, split='test')

    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_set)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data_set)

    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)
    test_batch_sampler = torch.utils.data.BatchSampler(test_sampler, args.batch_size, drop_last=True)

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True,
                                               num_workers=nw)
    test_loader = torch.utils.data.DataLoader(test_data_set,
                                              batch_sampler=test_batch_sampler,
                                              pin_memory=True,
                                              num_workers=nw)
    return train_sampler, test_sampler, train_loader, test_loader


def get_model(device, checkpoint_path, rank, args):
    model = PartNormalDataset()

    if rank == 0:
        torch.save(model.state_dict(), checkpoint_path)
    dist.barrier()

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    if args.syncBN:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    return model


def get_optimizer_and_scheduler(model, args):
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.005)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    return optimizer, scheduler


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        aug1, aug2, cls, _ = data
        loss = model(aug1, aug2, cls)
        loss = loss.mean()  # 将多个GPU返回的loss取平均
        loss.backward()
        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        # 在进程0中打印平均loss
        if is_main_process():
            data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return mean_loss.item()


def main_fn(rank, world_size, args):
    checkpoint_path, rank, device = init_process(rank, world_size, args)
    train_sampler, test_sampler, train_loader, test_loader = get_sampler_and_dataloader(rank, args)
    model = get_model(device, checkpoint_path, rank, args)
    optimizer, scheduler = get_optimizer_and_scheduler(model, args)

    if rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
        print(args)
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter()
        if os.path.exists("./weights_v9") is False:
            os.makedirs("./weights_v9")

    print('Start training...')

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        train_mean_loss = train_one_epoch(model=model,
                                          optimizer=optimizer,
                                          data_loader=train_loader,
                                          device=device,
                                          epoch=epoch)
        scheduler.step()

        # add evaluation
        test_sampler.set_epoch(epoch)
        test_mean_loss = evaluate(model=model,
                                  data_loader=test_loader,
                                  device=device)

        if rank == 0:
            tags = ["train_loss", "learning_rate", "eval_loss"]
            tb_writer.add_scalar(tags[0], train_mean_loss, epoch)
            tb_writer.add_scalar(tags[1], optimizer.param_groups[0]["lr"], epoch)
            tb_writer.add_scalar(tags[2], test_mean_loss, epoch)

            torch.save(model.module.state_dict(), "./weights_v9/model_knn_1024_fps-{}.pth".format(epoch))

    # 删除临时缓存文件
    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)
    cleanup()


def run(ws, terminal_parameters):
    processes = []
    for rank in range(ws):
        p = mp.Process(target=main_fn, args=(rank, ws, terminal_parameters))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights_v9')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.001)
    parser.add_argument('--syncBN', type=bool, default=True)
    parser.add_argument('--root', type=str, default='/mnt/longvideo/jinkun/4_liang/modelnet40_normal_resampled')
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--world_size', default=8, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    opt = parser.parse_args()
    world_size = opt.world_size
    run(world_size, opt)
