import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


@torch.no_grad()
def momentum_update(online, target, tao=0.99):
    if target is None:
        target = copy.deepcopy(online)
    else:
        for online_params, target_params in zip(online.parameters(), target.parameters()):
            target_weight, online_weight = target_params.data, online_params.data
            target_params.data = target_weight * tao + (1 - tao) * online_weight
    for parameter in target.parameters():
        parameter.requires_grad = False
    return target


class CrossedAttention(nn.Module):
    def __init__(self, channel=1024):
        super().__init__()
        self.q_conv = nn.Conv1d(channel, channel // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channel, channel // 4, 1, bias=False)
        self.v_conv = nn.Conv1d(channel, channel, 1, bias=False)
        self.trans_conv = nn.Conv1d(channel, channel, 1)
        self.after_norm = nn.BatchNorm1d(channel)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q_tensor, kv_tensor):

        x_q = self.q_conv(q_tensor.permute(0, 2, 1))
        x_k = self.k_conv(kv_tensor.permute(0, 2, 1))
        x_v = self.v_conv(kv_tensor.permute(0, 2, 1))

        energy = torch.matmul(x_q.permute(0, 2, 1), x_k)
        attention = self.softmax(energy)
        # attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.matmul(attention, x_v.permute(0, 2, 1))
        res = (q_tensor - x_r).permute(0, 2, 1)
        x_r = self.act(self.after_norm(self.trans_conv(res)))
        x_r = x_r.permute(0, 2, 1) + q_tensor

        return x_r


class EncoderPSG(nn.Module):
    def __init__(self, part_num=50):
        super(EncoderPSG, self).__init__()
        self.part_num = part_num
        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        # can change to SA_MH_Layer
        self.sa1 = SA_Layer(128)
        self.sa2 = SA_Layer(128)
        self.sa3 = SA_Layer(128)
        self.sa4 = SA_Layer(128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(0.2))

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(0.2))

        # modified here add conv3 and bn3
        self.conv3 = nn.Conv1d(1024 * 3 + 64, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()

    def forward(self, x, cls_label):
        x = x.permute(0, 2, 1)  # [B,N,3] -> [B,3,N]
        # print('x device: ', x.device)
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.concat((x1, x2, x3, x4), dim=1)
        x = self.conv_fuse(x)
        x_max = torch.max(x, 2)[0]  # [B, 1024]
        x_avg = x.mean(dim=2)  # [B, 1024]
        # print(x_max.shape)
        x_max_feature = x_max.reshape(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)  # [B, 1024, N]
        x_avg_feature = x_avg.reshape(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        # print('cls_label size:', cls_label.shape)
        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        # print(cls_label_one_hot.type())
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)
        x_global_feature = torch.concat((x_max_feature, x_avg_feature, cls_label_feature), 1) # 1024 + 64
        x = torch.concat((x, x_global_feature), 1)  # 1024 * 3 + 64
        x = self.bn3(self.conv3(x))  # [1024, N]
        x_max = x_max.reshape(batch_size, 1, -1)
        x_avg = x_avg.reshape(batch_size, 1, -1)
        x = x.permute(0, 2, 1)
        return x_max, x_avg, x


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)  # b, n, c
        x_k = self.k_conv(x)  # b, c, n
        x_v = self.v_conv(x)
        energy = torch.matmul(x_q, x_k)  # b, n, n
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.matmul(x_v, attention)  # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


class SA_MH_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1, bias=False)

        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        bs, f, p = x.shape
        x_q = self.q_conv(x)  # b, n, c
        x_q = x_q.reshape(bs, 4, -1, p).permute(0, 1, 3, 2)
        x_k = self.k_conv(x)
        x_k = x_k.reshape(bs, 4, -1, x.shape[-1])
        xy = torch.matmul(x_q, x_k)
        x_v = self.v_conv(x)
        x_v = x_v.reshape(bs, 4, -1, x.shape[-1]).permute(0, 1, 3, 2)
        xyz = torch.matmul(xy, x_v)
        xyz = xyz.permute(0, 1, 3, 2).reshape(bs, p, -1)
        xyz = self.trans_conv(xyz)
        xyz = self.act(self.after_norm(xyz - x))
        xyz = x + xyz
        return xyz


class SimSeg(nn.Module):
    def __init__(self):
        super(SimSeg, self).__init__()
        self.online_encoder = EncoderPSG().to(device)
        self.target_encoder = None
        self.online_x_attn = CrossedAttention().to(device)
        self.target_x_attn = None

    def forward(self, aug1, aug2, label):
        label = label.reshape(label.shape[0], -1)
        # print('label size: ', label.shape)
        # label = label.long()
        self.target_encoder = momentum_update(self.online_encoder, self.target_encoder)
        self.target_x_attn = momentum_update(self.online_x_attn, self.target_x_attn)

        # [B, 1, 1024], [B, 1, 1024], [B, N, 1024]
        f_max_1, f_avg_1, f_point_1 = self.online_encoder(aug1, label)
        f_max_2, f_avg_2, f_point_2 = self.target_encoder(aug2, label)
        # print('f_max size: ', f_max_1.shape)
        # print('f_avg size: ', f_avg_1.shape)
        # print('f_point size: ', f_point_1.shape)
        # max attention [B, 1, 1024]
        max_attn_1 = self.online_x_attn(f_max_1, f_point_1)
        max_attn_2 = self.target_x_attn(f_max_2, f_point_2)
        max_attn_3 = self.online_x_attn(f_max_2, f_point_2)
        max_attn_4 = self.target_x_attn(f_max_1, f_point_1)
        # print('max_attn size: ', max_attn_1.shape)
        max_loss_1 = loss_fn(max_attn_1, max_attn_2)
        max_loss_2 = loss_fn(max_attn_3, max_attn_4)

        # avg attention [B, 1, 1024]
        avg_attn_1 = self.online_x_attn(f_avg_1, f_point_1)
        avg_attn_2 = self.target_x_attn(f_avg_2, f_point_2)
        avg_attn_3 = self.online_x_attn(f_avg_2, f_point_2)
        avg_attn_4 = self.target_x_attn(f_avg_1, f_point_1)
        # print('avg_attn size: ', avg_attn_1.shape)
        avg_loss_1 = loss_fn(avg_attn_1, avg_attn_2)
        avg_loss_2 = loss_fn(avg_attn_3, avg_attn_4)

        loss = max_loss_1 + max_loss_2 + avg_loss_1 + avg_loss_2
        return loss.mean()


if __name__ == '__main__':
    rand_x_1 = torch.rand([4, 2679, 3])
    rand_x_2 = torch.rand([4, 2867, 3])
    rand_l = torch.rand([4, 16])
    # encoder = EncoderPSG()
    # x_attn = CrossedAttention()
    sim_seg = SimSeg()
    # seg_loss = sim_seg(rand_x_1, rand_x_2, rand_l)
    # f_max, f_avg, p = encoder(rand_x, rand_l)
    # print('f_max size: ', f_max.shape)
    # print('f_avg size: ', f_avg.shape)
    # print('p size: ', p.shape)

    # from psg_dataloader import PartNormalDataset
    # file_path = r'/home/haruki/下载/shapenet/shapenetcore'
    # dataset = PartNormalDataset(file_path)
    # Dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    # m1, m2, l = 0, 0, 0
    # for morph_1, morph_2, class_label, seg_label in Dataloader:
    #     m1 = morph_1
    #     m2 = morph_2
    #     l = class_label
    #     break
    #
    # print(m1.shape, m2.shape, l.shape)
    # seg_loss = sim_seg(m1, m2, l)
