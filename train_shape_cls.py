from network.encoder import PCT_Encoder
from utils.crops import *
from utils.provider import *
from dataloader import ModelNetDataSet
from torch.utils.data import DataLoader
from network.shape_classifier import ShapeClassifier
import logging
from tqdm import tqdm
import sys
from torch.utils.tensorboard import SummaryWriter
import os
import math


def test(model, loader, num_class=40):
    print("Start test process...")
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
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
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def cls_main():
    logger = logging.getLogger(__name__)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tb_writer = SummaryWriter()
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    root = '/home/akira/下载/Pointnet2_PyTorch-master/PCT/Point-Transformers-master/data/modelnet40_normal_resampled'
    train_dataset = ModelNetDataSet(root, split='train')
    test_dataset = ModelNetDataSet(root, split='test')
    BATCH_SIZE = 32
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    online_encoder = PCT_Encoder().to(device)
    eval_net = ShapeClassifier(net=online_encoder, sub_function=b_fps).to(device)

    model_save_path = r'/home/akira/下载/Pointnet2_PyTorch-master/SimAttention/scripts/weights/model-35-lr-x10.pth'
    loaded_paras = torch.load(model_save_path)
    saved_model = loaded_paras
    eval_net_dict = eval_net.state_dict()
    new_state_dict = {k: v for k, v in saved_model.items() if k in eval_net_dict.keys()}
    eval_net_dict.update(new_state_dict)
    eval_net.load_state_dict(eval_net_dict)

    classifier = eval_net
    criterion = torch.nn.CrossEntropyLoss()

    try:
        checkpoint = torch.load('best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        print('Use pretrain model')
    except:
        print('No existing model, starting training from scratch...')
        start_epoch = 0

    max_epochs = 50
    learning_rate = 0.01
    # todo: set more start lr and lrf

    pg = [p for p in classifier.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(pg, lr=learning_rate, momentum=0.9, weight_decay=0.005)
    lf = lambda x: ((1 + math.cos(x * math.pi / max_epochs)) / 2) * (1 - 0.01) + 0.01  # cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    best_epoch = 0
    mean_correct = []

    print('Start training from {} epoch'.format(start_epoch))
    for epoch in range(start_epoch, max_epochs):
        tags = ["instance_acc", "class_acc", "learning_rate", "ten_step_loss"]
        print('{} epoch is running'.format(epoch))
        classifier.train()
        ten_step_loss = 0.0
        trainDataLoader = tqdm(trainDataLoader, file=sys.stdout)
        for step, data in enumerate(trainDataLoader):
            points, target = data
            points = points.data.numpy()
            points = random_point_dropout(points)
            points[:, :, 0:3] = random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            target = target[:, 0]

            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()

            pred = classifier(points)
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
            loss.backward()
            optimizer.step()
            global_step += 1

        scheduler.step()

        train_instance_acc = np.mean(mean_correct)
        logger.info('Train Instance Accuracy: %f' % train_instance_acc)

        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader)

            if instance_acc >= best_instance_acc:
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if class_acc >= best_class_acc:
                best_class_acc = class_acc
            logger.info('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            logger.info('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if instance_acc >= best_instance_acc:
                logger.info('Save model...')
                save_path = 'best_model.pth'
                logger.info('Saving at %s' % save_path)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, save_path)
            global_epoch += 1

            tb_writer.add_scalar(tags[0], instance_acc, epoch)
            tb_writer.add_scalar(tags[1], class_acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        # torch.save(classifier.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    cls_main()
