import os
import math
import torch
import logging
import argparse
import tempfile
import torch.multiprocessing as mp
from utils.crops import *
from torch.utils.data import DataLoader
from model import SimAttention_5
from model import CrossedAttention
from torch.utils.tensorboard import SummaryWriter
from dataloader import ModelNetDataSet
from network.encoder import PCT_Encoder
from network.augmentation import Batch_PointWOLF
from utils.train_eval_utils import train_one_epoch
from utils.distributed_utils import init_distributed_mode, dist, cleanup


def main_fn(rank, world_size, args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    logger = logging.getLogger(__name__)

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
    # 初始化各进程环境 end

    rank = args.rank
    device = torch.device(args.device)
    batch_size = args.batch_size
    weights_path = args.weights
    args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增
    checkpoint_path = ""

    if rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
        print(args)
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter()
        if os.path.exists("./weights") is False:
            os.makedirs("./weights")

    # 实例化训练数据集
    train_data_set = ModelNetDataSet(args.root, split='train')

    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_set)

    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size, drop_last=True)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True,
                                               num_workers=nw)

    aug_method = Batch_PointWOLF()
    online_encoder = PCT_Encoder().cuda()
    crossed_method = CrossedAttention(1024).cuda()

    model = SimAttention_5(aug_method, b_fps, b_get_slice, b_get_cube, b_get_sphere,
                           online_encoder, crossed_method)

    checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
    # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
    if rank == 0:
        torch.save(model.state_dict(), checkpoint_path)
    dist.barrier()
    # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    if args.syncBN:
        # 使用SyncBatchNorm后训练会更耗时
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # optimizer
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=0.005)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    logger.info('Start training...')
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)
        scheduler.step()

        if rank == 0:
            tags = ["loss", "learning_rate"]
            tb_writer.add_scalar(tags[0], mean_loss, epoch)
            tb_writer.add_scalar(tags[1], optimizer.param_groups[0]["lr"], epoch)

            torch.save(model.module.state_dict(), "./weights/model-{}.pth".format(epoch))

    # 删除临时缓存文件
    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)
    cleanup()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=40)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)
    # 是否启用SyncBatchNorm
    parser.add_argument('--syncBN', type=bool, default=True)

    # 数据集所在根目录
    parser.add_argument('--root', type=str,
                        default='./data/modelnet40_normal_resampled')

    parser.add_argument('--freeze_layers', type=bool, default=False)
    # 不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world_size', default=2, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    opt = parser.parse_args()

    world_size = opt.world_size
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=main_fn, args=(rank, world_size, opt))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
