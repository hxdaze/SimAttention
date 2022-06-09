import os
import math
import torch
import argparse
import tempfile
import torch.multiprocessing as mp
from tqdm import tqdm
from utils.crops import *
from utils.provider import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataloader import ModelNetDataSet
from network.encoder import PCT_Encoder
from utils.distributed_utils import dist, cleanup
from network.shape_classifier import ShapeClassifier
from utils.distributed_utils import reduce_value, is_main_process
import sys
import logging


def test(model, loader, device, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    # 在进程0中打印验证进度
    if is_main_process():
        loader = tqdm(loader, file=sys.stdout)

    for j, data in enumerate(loader):
        points, target = data
        target = target[:, 0]
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]
        t = target.cpu().data.numpy()
        for cat in np.unique(t):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = torch.Tensor(np.mean(class_acc[:, 2])).to(device)
    instance_acc = torch.Tensor(np.mean(mean_correct)).to(device)

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    class_acc = reduce_value(class_acc, average=False)
    instance_acc = reduce_value(instance_acc, average=False)

    return instance_acc, class_acc


def train_one_cls_epoch(model, optimizer, scheduler, data_loader, test_loader, device, epoch, tb_writer):
    model.train()
    mean_loss = torch.zeros(1).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer.zero_grad()
    mean_correct = []
    best_instance_acc = 0.0
    best_class_acc = 0.0
    best_epoch = 0
    ten_step_loss = 0.0
    tags = ["instance_acc", "class_acc", "learning_rate", "ten_step_loss"]

    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)
        testDataLoader = tqdm(test_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        points, target = data

        points = points.data.numpy()
        points = random_point_dropout(points)
        points = random_scale_point_cloud(points)
        points = shift_point_cloud(points)
        points = torch.Tensor(points).cuda()
        target = target[:, 0].cuda()
        optimizer.zero_grad()
        pred = model(points)
        loss = criterion(pred, target.long())

        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

        if step > 0 and step % 10 == 0:
            print("{} step loss is: ".format(step), loss.item())
            ten_step_loss += loss.item()
            print('==========10 steps mean loss==========')
            print("{} step mean loss is: ".format(step), ten_step_loss / 10)
            ten_step_loss = 0.0
        else:
            ten_step_loss += loss.item()
            print("{} step loss is: ".format(step), loss.item())
        tb_writer.add_scalar(tags[3], ten_step_loss, step)

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

    scheduler.step()

    train_instance_acc = np.mean(mean_correct)
    print('Train Instance Accuracy: %f' % train_instance_acc)

    with torch.no_grad():
        instance_acc, class_acc = test(model.eval(), testDataLoader, device)

        if instance_acc >= best_instance_acc:
            best_instance_acc = instance_acc
            best_epoch = epoch + 1

        if class_acc >= best_class_acc:
            best_class_acc = class_acc
        print('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
        print('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

        if instance_acc >= best_instance_acc:
            print('Save model...')
            save_path = 'best_model.pth'
            print('Saving at %s' % save_path)
            state = {
                'epoch': best_epoch,
                'instance_acc': instance_acc,
                'class_acc': class_acc,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, save_path)

        tb_writer.add_scalar(tags[0], instance_acc, epoch)
        tb_writer.add_scalar(tags[1], class_acc, epoch)
    tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)


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
    train_dataset = ModelNetDataSet(args.root, split='train')
    test_dataset = ModelNetDataSet(args.root, split='test')
    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size, drop_last=True)
    test_batch_sampler = torch.utils.data.BatchSampler(
        test_sampler, batch_size, drop_last=True)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True,
                                               num_workers=nw)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_sampler=test_batch_sampler,
                                              pin_memory=True,
                                              num_workers=nw)

    online_encoder = PCT_Encoder().cuda()
    model = ShapeClassifier(net=online_encoder, sub_function=b_fps)

    # load pretrained encoder weights
    loaded_paras = torch.load(args.model_save_path)
    saved_model = loaded_paras
    model_dict = model.state_dict()
    new_state_dict = {k: v for k, v in saved_model.items() if k in model_dict.keys()}
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
    # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
    # 其实可以继续这样做，因为model也只是一部分encoder才有weights，其余的没有
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
        train_one_cls_epoch(model=model, optimizer=optimizer, scheduler=scheduler,
                            data_loader=train_loader, test_loader=test_loader, device=device, epoch=epoch,
                            tb_writer=tb_writer)

    # 删除临时缓存文件
    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)
    cleanup()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=40)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--model_save_path', type=str, default='./scripts/weights/model-35-lr-x10.pth')
    # 是否启用SyncBatchNorm
    parser.add_argument('--syncBN', type=bool, default=False)

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
