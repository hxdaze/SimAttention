import os
import torch
import numpy as np
from tqdm import tqdm
from utils.provider import *
from dataloader import ModelNetDataSet
from network.encoder import PCT_Encoder, Original_PCT_Encoder
from network.dgcnn_encoder import DGCNN_encoder


def make_file(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        print("New File established!", file_path)
    else:
        print("The File Already exists!", file_path)


def get_new_file_path(model_path):
    model_name = model_path[45:-4]
    print("The chosen model is: ", model_name)
    file = r'/home/haruki/下载/SimAttention/cls_data/' + model_name
    train_file = os.path.join(file, 'train')
    test_file = os.path.join(file, 'test')
    make_file(train_file)
    make_file(test_file)
    return train_file, test_file


def get_encoder(model_save_path):
    loaded_paras = torch.load(model_save_path)

    if model_save_path.endswith('t7'):
        encoder = Original_PCT_Encoder().to(device)
    elif 'dgcnn' in model_save_path:
        encoder = DGCNN_encoder().to(device)
    else:
        encoder = PCT_Encoder().to(device)

    encoder_dict = encoder.state_dict()

    new_state_dict = {}
    break_flag = 0

    for k in loaded_paras.keys():
        if k.startswith('online_encoder'):
            new_k = k[15:]
            new_state_dict[new_k] = loaded_paras[k]
        elif k.startswith('module'):
            break_flag += 1
            new_k = k[7:]
            new_state_dict[new_k] = loaded_paras[k]
            if break_flag == 98:
                break

    encoder_dict.update(new_state_dict)
    encoder.load_state_dict(encoder_dict)
    return encoder


def get_data_loader(root, tag, is_dgcnn):
    # tag = 'train' / 'test'
    data_set = ModelNetDataSet(root, split=tag)
    # dgcnn = 8 failed!
    if is_dgcnn:
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=True)
    else:
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=8, shuffle=True)
    return tqdm(data_loader)


def save_txt(x, label, file_path, step):
    file_name = os.path.join(file_path, str(step) + '.txt')
    l = torch.unsqueeze(label, 1)  # [B] ---> [B, 1]
    data = torch.cat((x, l), 1)  # [B, 1024] ---> [B, 1025]
    data = data.detach().cpu().numpy()  # [B, 1025]
    np.savetxt(file_name, data, delimiter=',')


def get_latent_representation(loader, encoder, storage_path):
    for step, data in enumerate(loader):
        points, target = data
        points = points.data.numpy()
        points = random_point_dropout(points)
        points[:, :, 0:3] = random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = shift_point_cloud(points[:, :, 0:3])
        points = torch.Tensor(points)
        target = target[:, 0]  # [B]
        points, target = points.to(device), target.to(device)
        get_global_feature = encoder.eval()
        latent_rep = get_global_feature(points).reshape(points.shape[0], -1)
        save_txt(latent_rep, target, storage_path, step)
        # print("New Latent Representation Dataset Established!")


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # original dataset file path
    root = '/home/haruki/下载/modelnet40_normal_resampled'
    name_list = [10, 15, 20, 25, 30, 40]
    # pretrained model name
    for name_index in name_list:
        # next: model_knn_2048_0.4_0.8_16_an_-
        model_name = 'model_knn_2048_2_0.4_0.8_8_an_-' + str(name_index)
        # pretrained weights path
        model_save_path = os.path.join("/home/haruki/下载/SimAttention/scripts/weights", model_name + '.pth')
        use_dgcnn = False
        # different batch size setup
        if 'dgcnn' in model_save_path:
            use_dgcnn = True

        train_file_path, test_file_path = get_new_file_path(model_path=model_save_path)
        net_encoder = get_encoder(model_save_path=model_save_path)
        train_loader = get_data_loader(root=root, tag='train', is_dgcnn=use_dgcnn)
        test_loader = get_data_loader(root=root, tag='test', is_dgcnn=use_dgcnn)

        get_latent_representation(train_loader, net_encoder, train_file_path)
        print("Train Dataset Done!")
        get_latent_representation(test_loader, net_encoder, test_file_path)
        print("Test Dataset Done!")
