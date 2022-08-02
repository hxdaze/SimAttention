import os
import math
import torch
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from dataloader import LatentRepresentationDataSet
from network.shape_classifier_2 import ShapeClassifier_2


# train and test data loader
def get_loader(root):
    train_root = os.path.join(root, 'train')
    test_root = os.path.join(root, 'test')
    train_lrds = LatentRepresentationDataSet(train_root)
    test_lrds = LatentRepresentationDataSet(test_root)
    train_loader = torch.utils.data.DataLoader(train_lrds, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_lrds, batch_size=1, shuffle=True)
    # delete the tqdm function
    return train_loader, test_loader


# log file
def log_func(log_file):
    if not os.path.exists(log_file):
        os.makedirs(log_file)
        print("Make Log File!")
    else:
        print("Log File Already Exists")
    tensorboard_writer = SummaryWriter(log_dir=log_file)
    return tensorboard_writer


# evaluate classifier on test dataset
@torch.no_grad()
def evaluate_model(model, loader, device):
    test_mean_correct = []
    test_loss = 0.0
    for f, l in loader:
        f, l = f.to(device), l.to(device)
        f = f.reshape(-1, 1024)
        l = l.reshape(l.shape[-1] * l.shape[0])  # torch.Size([8])
        cls = model.eval()
        pred = cls(f.float())
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(pred, l.long())
        test_loss += loss.item()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(l.long().data).cpu().sum()
        test_mean_correct.append(correct.item() / float(f.size()[0]))
    test_instance_acc = np.mean(test_mean_correct)
    return test_instance_acc, test_loss / len(loader)


# major function
def train_and_test(device, tb_writer, trainDataLoader, testDataLoader, args):
    learning_rate = args.lr
    learning_rate_final = args.lrf
    max_epochs = args.epoch
    opt_choice = args.opt_choice
    classifier = ShapeClassifier_2().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    if opt_choice == 0:
        optimizer = torch.optim.SGD(classifier.parameters(),
                                    lr=learning_rate,
                                    momentum=0.9,
                                    weight_decay=0.005)
    if opt_choice == 1:
        # new optimizer AdamW
        optimizer = torch.optim.AdamW(classifier.parameters(),
                                      lr=learning_rate,
                                      betas=(0.9, 0.999),
                                      weight_decay=0.01)
    lf = lambda x: ((1 + math.cos(x * math.pi / max_epochs)) / 2) * (1 - learning_rate_final) + learning_rate_final
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    tags = ["train_acc", "learning_rate", "train_loss", "test_acc", "test_loss"]

    # record the best information
    best_test_acc = 0.0
    pair_train_acc = 0.0
    best_test_epoch = 0

    for epoch in range(0, max_epochs):
        # train mean correct
        mean_correct = []
        train_loss = 0.0
        for f, l in trainDataLoader:
            f, l = f.to(device), l.to(device)
            f = f.reshape(-1, 1024)  # torch.Size([8, 1024])
            l = l.reshape(l.shape[-1] * l.shape[0])  # torch.Size([8])

            optimizer.zero_grad()
            pred = classifier(f.float())  # torch.Size([8, 40])
            loss = criterion(pred, l.long())
            train_loss += loss.item()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(l.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(f.size()[0]))
            loss.backward()
            optimizer.step()
        scheduler.step()
        train_instance_acc = np.mean(mean_correct)

        tb_writer.add_scalar(tags[0], train_instance_acc, epoch)
        tb_writer.add_scalar(tags[1], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar(tags[2], train_loss / len(trainDataLoader), epoch)

        # evaluation part
        test_acc, test_loss = evaluate_model(classifier.eval(), testDataLoader, device)
        tb_writer.add_scalar(tags[3], test_acc, epoch)
        tb_writer.add_scalar(tags[4], test_loss, epoch)

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            pair_train_acc = train_instance_acc
            best_test_epoch = epoch
    print("For this model, best test_acc is {}, corresponding train_acc is {} and epoch is {}".format(best_test_acc, pair_train_acc, best_test_epoch))


def run(args):
    print("Chosen model is: ", args.data_file[38:])
    log_file = os.path.join(args.data_file, args.run_file_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_data_loader, test_data_loader = get_loader(args.data_file)
    train_and_test(device,
                   log_func(log_file),
                   train_data_loader,
                   test_data_loader,
                   args)
    # direct see the results without tensorboard command
    # command_str = 'tensorboard --logdir={}'.format(log_file)
    # os.system(command_str)


if __name__ == '__main__':
    dataset_name = 'model_knn_2048_0.4_0.8_16_an-'
    name_list = [10, 20, 30, 40]
    for name_index in name_list:
        dataset_name += str(name_index)
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_file', type=str,
                            default=os.path.join('/home/haruki/下载/SimAttention/cls_data/', dataset_name))
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--lrf', type=float, default=0.01)
        parser.add_argument('--epoch', type=int, default=100)
        parser.add_argument('--opt_choice', type=int, default=1)
        parser.add_argument('--run_file_name', type=str, default='run_adamw_100')
        opt = parser.parse_args()

        run(opt)
        if name_index < 10:
            dataset_name = dataset_name[:-1]
        else:
            dataset_name = dataset_name[:-2]
