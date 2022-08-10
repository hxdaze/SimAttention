import torch
import copy
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# loss fn from BYOL
def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


# project function from BYOL
class ProjectMLP(nn.Module):
    def __init__(self, input_dim=1024, output_dim=512, hidden_size=2048):
        super(ProjectMLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.l1 = nn.Linear(input_dim, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.bn(self.l1(x.reshape(x.shape[0], -1)))
        x = self.l2(self.relu(x))
        return x.reshape(x.shape[0], 1, -1)

    
# simsam project function
class ProjectMLP2(nn.Module):
    def __init__(self, input_dim=1024, output_dim=1024, hidden_size=4096):
        super(ProjectMLP2, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size

        self.l1 = nn.Linear(input_dim, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU(inplace=True)

        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

        self.l3 = nn.Linear(hidden_size, output_dim)
        self.bn3 = nn.BatchNorm1d(output_dim)


    def forward(self, x):
        x = self.bn1(self.l1(x.reshape(x.shape[0], -1)))
        x = self.bn2(self.l2(self.relu(x)))
        x = self.bn3(self.l3(self.relu(x)))
        return x.reshape(x.shape[0], 1, -1)

# crossed attention method:
# todo: method_1: train 2 different CAs
# method_2: momentum update the target_CA
# 50 epoch
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


# SimAttention Class
# NS1 without MLP projector
class SimAttention_1(nn.Module):
    def __init__(self,
                 aug_function,
                 sub_function,
                 slice_function,
                 cube_function,
                 sphere_function,
                 online_encoder,
                 attention_feature_method):
        super().__init__()
        self.aug_function = aug_function
        self.sub_function = sub_function
        self.slice_function = slice_function
        self.cube_function = cube_function
        self.sphere_function = sphere_function
        self.online_encoder = online_encoder
        self.target_encoder = None
        self.attention_feature = attention_feature_method

    def forward(self, x):
        x = x.cpu().numpy()
        aug1, aug2 = self.aug_function(x), self.aug_function(x)

        sub1, sub2 = torch.Tensor(self.sub_function(aug1, 1024)).to(device), torch.Tensor(
            self.sub_function(aug2, 1024)).to(device)
        slice1, slice2 = torch.Tensor(self.slice_function(aug1, 1, 1, 1024)).to(device), torch.Tensor(
            self.slice_function(aug2, 1, 1, 1024)).to(device)
        cube1, cube2 = torch.Tensor(self.cube_function(aug1, 0.2, 1024)).to(device), torch.Tensor(
            self.cube_function(aug2, 0.2, 1024)).to(device)
        sphere1, sphere2 = torch.Tensor(self.sphere_function(aug1, 0.2, 1024)).to(device), torch.Tensor(
            self.sphere_function(aug2, 0.1, 1024)).to(device)

        # [B, 1, N_f] N_f: output dimension of encoder
        sub_feature_1 = self.online_encoder(sub1)
        sub_feature_3 = self.online_encoder(sub2)
        # with torch.no_grad():
        #     self.target_encoder = copy.deepcopy(self.online_encoder)
        #     for parameter in self.target_encoder.parameters():
        #         parameter.requires_grad = False
        #     sub_feature_2 = self.target_encoder(sub2)
        #     sub_feature_4 = self.target_encoder(sub1)

        # with momentum encoder
        with torch.no_grad():
            if self.target_encoder is None:
                self.target_encoder = copy.deepcopy(self.online_encoder)
            else:
                for online_params, target_params in zip(self.online_encoder.parameters(),
                                                        self.target_encoder.parameters()):
                    target_weight, online_weight = target_params.data, online_params.data
                    # moving average decay is tao
                    tao = 0.99
                    target_params.data = target_weight * tao + (1 - tao) * online_weight
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad = False
            sub_feature_2 = self.target_encoder(sub2)
            sub_feature_4 = self.target_encoder(sub1)

        # slice feature [B, 1, N_f]
        slice_feature_1 = self.online_encoder(slice1)
        slice_feature_2 = self.online_encoder(slice2)

        # cube feature  [B, 1, N_f]
        cube_feature_1 = self.online_encoder(cube1)
        cube_feature_2 = self.online_encoder(cube2)

        # sphere feature [B, 1, N_f]
        sphere_feature_1 = self.online_encoder(sphere1)
        sphere_feature_2 = self.online_encoder(sphere2)

        # crop feature concat [B, 3, N_f]
        crop_feature_1 = torch.cat((slice_feature_1, cube_feature_1, sphere_feature_1), dim=1)
        crop_feature_2 = torch.cat((slice_feature_2, cube_feature_2, sphere_feature_2), dim=1)
        # [B, 6, N_f]
        crop_feature = torch.cat((crop_feature_1, crop_feature_2), dim=1)

        # attention feature
        # todo: crossed attention module, cross_attention func name
        attn_feature_1 = self.attention_feature(sub_feature_1, crop_feature)
        attn_feature_2 = self.attention_feature(sub_feature_2, crop_feature)

        attn_feature_3 = self.attention_feature(sub_feature_3, crop_feature)
        attn_feature_4 = self.attention_feature(sub_feature_4, crop_feature)

        # loss
        loss_1 = loss_fn(attn_feature_1, attn_feature_2)
        loss_2 = loss_fn(attn_feature_3, attn_feature_4)
        loss = loss_1 + loss_2

        # return loss.mean()
        return loss.mean()


# NS2
class SimAttention_2(nn.Module):
    def __init__(self,
                 aug_function,
                 sub_function,
                 slice_function,
                 cube_function,
                 sphere_function,
                 online_encoder,
                 attention_feature_method):
        super().__init__()
        self.aug_function = aug_function
        self.sub_function = sub_function
        self.slice_function = slice_function
        self.cube_function = cube_function
        self.sphere_function = sphere_function
        self.online_encoder = online_encoder
        self.target_encoder = None
        self.attention_feature = attention_feature_method

    def forward(self, x):
        x = x.cpu().numpy()
        aug1, aug2 = self.aug_function(x), self.aug_function(x)

        sub1, sub2 = torch.Tensor(self.sub_function(aug1, 1024)).to(device), torch.Tensor(
            self.sub_function(aug2, 1024)).to(device)
        slice1, slice2 = torch.Tensor(self.slice_function(aug1, 1, 1, 1024)).to(device), torch.Tensor(
            self.slice_function(aug2, 1, 1, 1024)).to(device)
        cube1, cube2 = torch.Tensor(self.cube_function(aug1, 0.2, 1024)).to(device), torch.Tensor(
            self.cube_function(aug2, 0.2, 1024)).to(device)
        sphere1, sphere2 = torch.Tensor(self.sphere_function(aug1, 0.2, 1024)).to(device), torch.Tensor(
            self.sphere_function(aug2, 0.1, 1024)).to(device)

        # [B, 1, N_f] N_f: output dimension of encoder
        sub_feature_1 = self.online_encoder(sub1)
        sub_feature_3 = self.online_encoder(sub2)
        with torch.no_grad():
            if self.target_encoder is None:
                self.target_encoder = copy.deepcopy(self.online_encoder)
            else:
                for online_params, target_params in zip(self.online_encoder.parameters(),
                                                        self.target_encoder.parameters()):
                    target_weight, online_weight = target_params.data, online_params.data
                    # moving average decay is tao
                    tao = 0.99
                    target_params.data = target_weight * tao + (1 - tao) * online_weight
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad = False
            sub_feature_2 = self.target_encoder(sub2)
            sub_feature_4 = self.target_encoder(sub1)

        # slice feature [B, 1, N_f]
        slice_feature_1 = self.online_encoder(slice1)
        slice_feature_2 = self.online_encoder(slice2)

        # cube feature  [B, 1, N_f]
        cube_feature_1 = self.online_encoder(cube1)
        cube_feature_2 = self.online_encoder(cube2)

        # sphere feature [B, 1, N_f]
        sphere_feature_1 = self.online_encoder(sphere1)
        sphere_feature_2 = self.online_encoder(sphere2)

        # crop feature concat [B, 3, N_f]
        crop_feature_1 = torch.cat((slice_feature_1, cube_feature_1, sphere_feature_1), dim=1)
        crop_feature_2 = torch.cat((slice_feature_2, cube_feature_2, sphere_feature_2), dim=1)
        # [B, 6, N_f]
        crop_feature = torch.cat((crop_feature_1, crop_feature_2), dim=1)

        # attention feature
        attn_feature_1 = self.attention_feature(sub_feature_1, crop_feature)
        attn_feature_3 = self.attention_feature(sub_feature_3, crop_feature)

        # loss
        print('attn_feature shape: ', attn_feature_1.shape)
        print('sub_feature shape: ', sub_feature_2.shape)
        loss_1 = loss_fn(attn_feature_1, sub_feature_2.reshape(attn_feature_1.shape[0], -1))
        print('sub_feature shape: ', sub_feature_2.shape)
        print('loss shape: ', loss_1.shape)
        loss_2 = loss_fn(attn_feature_3, sub_feature_4)
        loss = loss_1 + loss_2

        return loss


# NS3
class SimAttention_3(nn.Module):
    def __init__(self,
                 aug_function,
                 sub_function,
                 slice_function,
                 cube_function,
                 sphere_function,
                 online_encoder,
                 attention_feature_method):
        super().__init__()
        self.aug_function = aug_function
        self.sub_function = sub_function
        self.slice_function = slice_function
        self.cube_function = cube_function
        self.sphere_function = sphere_function
        self.online_encoder = online_encoder
        self.target_encoder = None
        self.attention_feature = attention_feature_method

    def forward(self, x):
        x = x.cpu().numpy()
        aug1, aug2 = self.aug_function(x), self.aug_function(x)

        sub1, sub2 = torch.Tensor(self.sub_function(aug1, 1024)).to(device), torch.Tensor(
            self.sub_function(aug2, 1024)).to(device)
        slice1, slice2 = torch.Tensor(self.slice_function(aug1, 1, 1, 1024)).to(device), torch.Tensor(
            self.slice_function(aug2, 1, 1, 1024)).to(device)
        cube1, cube2 = torch.Tensor(self.cube_function(aug1, 0.2, 1024)).to(device), torch.Tensor(
            self.cube_function(aug2, 0.2, 1024)).to(device)
        sphere1, sphere2 = torch.Tensor(self.sphere_function(aug1, 0.2, 1024)).to(device), torch.Tensor(
            self.sphere_function(aug2, 0.1, 1024)).to(device)

        # [B, 1, N_f] N_f: output dimension of encoder
        sub_feature_1 = self.online_encoder(sub1)
        sub_feature_3 = self.online_encoder(sub2)
        with torch.no_grad():
            if self.target_encoder is None:
                self.target_encoder = copy.deepcopy(self.online_encoder)
            else:
                for online_params, target_params in zip(self.online_encoder.parameters(),
                                                        self.target_encoder.parameters()):
                    target_weight, online_weight = target_params.data, online_params.data
                    # moving average decay is tao
                    tao = 0.99
                    target_params.data = target_weight * tao + (1 - tao) * online_weight
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad = False
            sub_feature_2 = self.target_encoder(sub2)
            sub_feature_4 = self.target_encoder(sub1)

        # slice feature [B, 1, N_f]
        slice_feature_1 = self.online_encoder(slice1)
        slice_feature_2 = self.online_encoder(slice2)

        # cube feature  [B, 1, N_f]
        cube_feature_1 = self.online_encoder(cube1)
        cube_feature_2 = self.online_encoder(cube2)

        # sphere feature [B, 1, N_f]
        sphere_feature_1 = self.online_encoder(sphere1)
        sphere_feature_2 = self.online_encoder(sphere2)

        # crop feature concat [B, 3, N_f]
        crop_feature_1 = torch.cat((slice_feature_1, cube_feature_1, sphere_feature_1), dim=1)
        crop_feature_2 = torch.cat((slice_feature_2, cube_feature_2, sphere_feature_2), dim=1)
        # [B, 6, N_f]
        # crop_feature = torch.cat((crop_feature_1, crop_feature_2), dim=1)

        # attention feature
        attn_feature_1 = self.attention_feature(sub_feature_1, crop_feature_2)
        attn_feature_2 = self.attention_feature(sub_feature_2, crop_feature_1)

        attn_feature_3 = self.attention_feature(sub_feature_3, crop_feature_2)
        attn_feature_4 = self.attention_feature(sub_feature_4, crop_feature_1)

        # loss
        loss_1 = loss_fn(attn_feature_1, attn_feature_2)
        loss_2 = loss_fn(attn_feature_3, attn_feature_4)
        loss = loss_1 + loss_2

        return loss


# NS1 with mlp projector
class SimAttention_4(nn.Module):
    def __init__(self,
                 aug_function,
                 sub_function,
                 slice_function,
                 cube_function,
                 sphere_function,
                 online_encoder,
                 project_method,
                 crossed_attention_method):
        super().__init__()
        self.aug_function = aug_function
        self.sub_function = sub_function
        self.slice_function = slice_function
        self.cube_function = cube_function
        self.sphere_function = sphere_function
        self.online_encoder = online_encoder
        self.target_encoder = None
        self.project_method = project_method
        self.crossed_attention = crossed_attention_method

    def forward(self, x):
        x = x.cpu().numpy()
        aug1, aug2 = self.aug_function(x), self.aug_function(x)

        sub1, sub2 = torch.Tensor(self.sub_function(aug1, 1024)).to(device), torch.Tensor(
            self.sub_function(aug2, 1024)).to(device)
        slice1, slice2 = torch.Tensor(self.slice_function(aug1, 1, 1, 1024)).to(device), torch.Tensor(
            self.slice_function(aug2, 1, 1, 1024)).to(device)
        cube1, cube2 = torch.Tensor(self.cube_function(aug1, 0.2, 1024)).to(device), torch.Tensor(
            self.cube_function(aug2, 0.2, 1024)).to(device)
        sphere1, sphere2 = torch.Tensor(self.sphere_function(aug1, 0.2, 1024)).to(device), torch.Tensor(
            self.sphere_function(aug2, 0.1, 1024)).to(device)

        # [B, 1, N_f] N_f: output dimension of mlp: 512
        sub_feature_1 = self.project_method(self.online_encoder(sub1))
        sub_feature_3 = self.project_method(self.online_encoder(sub2))

        # with momentum encoder
        with torch.no_grad():
            if self.target_encoder is None:
                self.target_encoder = copy.deepcopy(self.online_encoder)
            else:
                for online_params, target_params in zip(self.online_encoder.parameters(),
                                                        self.target_encoder.parameters()):
                    target_weight, online_weight = target_params.data, online_params.data
                    # moving average decay is tao
                    tao = 0.99
                    target_params.data = target_weight * tao + (1 - tao) * online_weight
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad = False
            sub_feature_2 = self.project_method(self.target_encoder(sub2))
            sub_feature_4 = self.project_method(self.target_encoder(sub1))

        # slice feature [B, 1, N_f]
        slice_feature_1 = self.project_method(self.online_encoder(slice1))
        slice_feature_2 = self.project_method(self.online_encoder(slice2))

        # cube feature  [B, 1, N_f]
        cube_feature_1 = self.project_method(self.online_encoder(cube1))
        cube_feature_2 = self.project_method(self.online_encoder(cube2))

        # sphere feature [B, 1, N_f]
        sphere_feature_1 = self.project_method(self.online_encoder(sphere1))
        sphere_feature_2 = self.project_method(self.online_encoder(sphere2))

        # crop feature concat [B, 3, N_f]
        crop_feature_1 = torch.cat((slice_feature_1, cube_feature_1, sphere_feature_1), dim=1)
        crop_feature_2 = torch.cat((slice_feature_2, cube_feature_2, sphere_feature_2), dim=1)
        # [B, 6, N_f]
        crop_feature = torch.cat((crop_feature_1, crop_feature_2), dim=1)

        # attention feature
        attn_feature_1 = self.crossed_attention(sub_feature_1, crop_feature)
        attn_feature_2 = self.crossed_attention(sub_feature_2, crop_feature)
        attn_feature_3 = self.crossed_attention(sub_feature_3, crop_feature)
        attn_feature_4 = self.crossed_attention(sub_feature_4, crop_feature)

        # loss
        loss_1 = loss_fn(attn_feature_1, attn_feature_2)
        loss_2 = loss_fn(attn_feature_3, attn_feature_4)
        loss = loss_1 + loss_2

        return loss.mean()


# without mlp
class SimAttention_5(nn.Module):
    def __init__(self,
                 aug_function,
                 sub_function,
                 slice_function,
                 cube_function,
                 sphere_function,
                 online_encoder,
                 crossed_attention_method):
        super().__init__()
        self.aug_function = aug_function
        self.sub_function = sub_function
        self.slice_function = slice_function
        self.cube_function = cube_function
        self.sphere_function = sphere_function
        self.online_encoder = online_encoder
        self.target_encoder = None

        self.crossed_attention = crossed_attention_method

    def forward(self, x):
        x = x.cpu().numpy()
        aug1, aug2 = self.aug_function(x), self.aug_function(x)

        sub1, sub2 = torch.Tensor(self.sub_function(aug1, 1024)).to(device), torch.Tensor(
            self.sub_function(aug2, 1024)).to(device)
        slice1, slice2 = torch.Tensor(self.slice_function(aug1, 1, 1, 1024)).to(device), torch.Tensor(
            self.slice_function(aug2, 1, 1, 1024)).to(device)
        cube1, cube2 = torch.Tensor(self.cube_function(aug1, 0.2, 1024)).to(device), torch.Tensor(
            self.cube_function(aug2, 0.2, 1024)).to(device)
        sphere1, sphere2 = torch.Tensor(self.sphere_function(aug1, 0.2, 1024)).to(device), torch.Tensor(
            self.sphere_function(aug2, 0.1, 1024)).to(device)

        # [B, 1, N_f] N_f: output dimension of mlp: 512
        sub_feature_1 = self.online_encoder(sub1)
        sub_feature_3 = self.online_encoder(sub2)

        # with momentum encoder
        with torch.no_grad():
            if self.target_encoder is None:
                self.target_encoder = copy.deepcopy(self.online_encoder)
            else:
                for online_params, target_params in zip(self.online_encoder.parameters(),
                                                        self.target_encoder.parameters()):
                    target_weight, online_weight = target_params.data, online_params.data
                    # moving average decay is tao
                    tao = 0.99
                    target_params.data = target_weight * tao + (1 - tao) * online_weight
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad = False
            sub_feature_2 = self.target_encoder(sub2)
            sub_feature_4 = self.target_encoder(sub1)

        # slice feature [B, 1, N_f]
        slice_feature_1 = self.online_encoder(slice1)
        slice_feature_2 = self.online_encoder(slice2)

        # cube feature  [B, 1, N_f]
        cube_feature_1 = self.online_encoder(cube1)
        cube_feature_2 = self.online_encoder(cube2)

        # sphere feature [B, 1, N_f]
        sphere_feature_1 = self.online_encoder(sphere1)
        sphere_feature_2 = self.online_encoder(sphere2)

        # crop feature concat [B, 3, N_f]
        crop_feature_1 = torch.cat((slice_feature_1, cube_feature_1, sphere_feature_1), dim=1)
        crop_feature_2 = torch.cat((slice_feature_2, cube_feature_2, sphere_feature_2), dim=1)
        # [B, 6, N_f]
        crop_feature = torch.cat((crop_feature_1, crop_feature_2), dim=1)

        # attention feature
        attn_feature_1 = self.crossed_attention(sub_feature_1, crop_feature)
        attn_feature_2 = self.crossed_attention(sub_feature_2, crop_feature)
        attn_feature_3 = self.crossed_attention(sub_feature_3, crop_feature)
        attn_feature_4 = self.crossed_attention(sub_feature_4, crop_feature)

        # loss
        loss_1 = loss_fn(attn_feature_1, attn_feature_2)
        loss_2 = loss_fn(attn_feature_3, attn_feature_4)
        loss = loss_1 + loss_2

        return loss.mean()


# NS1 without mlp, with tensor methods
class SimAttention_6(nn.Module):
    def __init__(self,
                 aug_function,
                 sub_function,
                 slice_function,
                 cube_function,
                 sphere_function,
                 online_encoder,
                 crossed_attention_method):
        super().__init__()
        self.aug_function = aug_function
        self.sub_function = sub_function
        self.slice_function = slice_function
        self.cube_function = cube_function
        self.sphere_function = sphere_function
        self.online_encoder = online_encoder
        self.target_encoder = None

        self.crossed_attention = crossed_attention_method

    def forward(self, x):
        x = x.to(device)
        aug1, aug2 = self.aug_function(x), self.aug_function(x)

        sub1, sub2 = self.sub_function(aug1, 1024), self.sub_function(aug2, 1024)
        slice1, slice2 = self.slice_function(aug1, 1, 1, 1024), self.slice_function(aug2, 1, 1, 1024)
        cube1, cube2 = self.cube_function(aug1, 0.2, 1024), self.cube_function(aug2, 0.2, 1024)
        sphere1, sphere2 = self.sphere_function(aug1, 0.2, 1024), self.sphere_function(aug2, 0.1, 1024)

        # [B, 1, N_f] N_f: output dimension of mlp: 512
        sub_feature_1 = self.online_encoder(sub1)
        sub_feature_3 = self.online_encoder(sub2)

        # with momentum encoder
        with torch.no_grad():
            if self.target_encoder is None:
                self.target_encoder = copy.deepcopy(self.online_encoder)
            else:
                for online_params, target_params in zip(self.online_encoder.parameters(),
                                                        self.target_encoder.parameters()):
                    target_weight, online_weight = target_params.data, online_params.data
                    # moving average decay is tao
                    tao = 0.99
                    target_params.data = target_weight * tao + (1 - tao) * online_weight
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad = False
            sub_feature_2 = self.target_encoder(sub2)
            sub_feature_4 = self.target_encoder(sub1)

        # slice feature [B, 1, N_f]
        slice_feature_1 = self.online_encoder(slice1)
        slice_feature_2 = self.online_encoder(slice2)

        # cube feature  [B, 1, N_f]
        cube_feature_1 = self.online_encoder(cube1)
        cube_feature_2 = self.online_encoder(cube2)

        # sphere feature [B, 1, N_f]
        sphere_feature_1 = self.online_encoder(sphere1)
        sphere_feature_2 = self.online_encoder(sphere2)

        # crop feature concat [B, 3, N_f]
        crop_feature_1 = torch.cat((slice_feature_1, cube_feature_1, sphere_feature_1), dim=1)
        crop_feature_2 = torch.cat((slice_feature_2, cube_feature_2, sphere_feature_2), dim=1)
        # [B, 6, N_f]
        crop_feature = torch.cat((crop_feature_1, crop_feature_2), dim=1)

        # attention feature
        attn_feature_1 = self.crossed_attention(sub_feature_1, crop_feature)
        attn_feature_2 = self.crossed_attention(sub_feature_2, crop_feature)
        attn_feature_3 = self.crossed_attention(sub_feature_3, crop_feature)
        attn_feature_4 = self.crossed_attention(sub_feature_4, crop_feature)

        # loss
        loss_1 = loss_fn(attn_feature_1, attn_feature_2)
        loss_2 = loss_fn(attn_feature_3, attn_feature_4)
        loss = loss_1 + loss_2

        return loss.mean()


# NS1 with online-target-x-attention methods
# with online and target crossed attention
class SimAttention_7(nn.Module):
    def __init__(self,
                 aug_function,
                 sub_function,
                 slice_function,
                 cube_function,
                 sphere_function,
                 online_encoder,
                 crossed_attention_method):
        super().__init__()
        self.aug_function = aug_function
        self.sub_function = sub_function
        self.slice_function = slice_function
        self.cube_function = cube_function
        self.sphere_function = sphere_function

        self.online_encoder = online_encoder
        self.target_encoder = None

        self.online_x_attn = crossed_attention_method
        self.target_x_attn = None

    def forward(self, x):
        x = x.cpu().numpy()
        aug1, aug2 = self.aug_function(x), self.aug_function(x)

        sub1, sub2 = torch.Tensor(self.sub_function(aug1, 1024)).to(device), torch.Tensor(
            self.sub_function(aug2, 1024)).to(device)
        slice1, slice2 = torch.Tensor(self.slice_function(aug1, 1, 1, 1024)).to(device), torch.Tensor(
            self.slice_function(aug2, 1, 1, 1024)).to(device)
        cube1, cube2 = torch.Tensor(self.cube_function(aug1, 0.2, 1024)).to(device), torch.Tensor(
            self.cube_function(aug2, 0.2, 1024)).to(device)
        sphere1, sphere2 = torch.Tensor(self.sphere_function(aug1, 0.2, 1024)).to(device), torch.Tensor(
            self.sphere_function(aug2, 0.1, 1024)).to(device)

        # [B, 1, N_f] N_f: output dimension of mlp: 512
        sub_feature_1 = self.online_encoder(sub1)
        sub_feature_3 = self.online_encoder(sub2)

        # with momentum encoder
        with torch.no_grad():
            if self.target_encoder is None:
                self.target_encoder = copy.deepcopy(self.online_encoder)
            else:
                for online_params, target_params in zip(self.online_encoder.parameters(),
                                                        self.target_encoder.parameters()):
                    target_weight, online_weight = target_params.data, online_params.data
                    # moving average decay is tao
                    tao = 0.99
                    target_params.data = target_weight * tao + (1 - tao) * online_weight
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad = False
            sub_feature_2 = self.target_encoder(sub2)
            sub_feature_4 = self.target_encoder(sub1)

        # slice feature [B, 1, N_f]
        slice_feature_1 = self.online_encoder(slice1)
        slice_feature_2 = self.online_encoder(slice2)

        # cube feature  [B, 1, N_f]
        cube_feature_1 = self.online_encoder(cube1)
        cube_feature_2 = self.online_encoder(cube2)

        # sphere feature [B, 1, N_f]
        sphere_feature_1 = self.online_encoder(sphere1)
        sphere_feature_2 = self.online_encoder(sphere2)

        # crop feature concat [B, 3, N_f]
        crop_feature_1 = torch.cat((slice_feature_1, cube_feature_1, sphere_feature_1), dim=1)
        crop_feature_2 = torch.cat((slice_feature_2, cube_feature_2, sphere_feature_2), dim=1)
        # [B, 6, N_f]
        crop_feature = torch.cat((crop_feature_1, crop_feature_2), dim=1)

        # attention feature
        with torch.no_grad():
            if self.target_x_attn is None:
                self.target_x_attn = copy.deepcopy(self.online_x_attn)
            else:
                for online_params, target_params in zip(self.online_x_attn.parameters(),
                                                        self.target_x_attn.parameters()):
                    target_weight, online_weight = target_params.data, online_params.data
                    # moving average decay is tao
                    tao = 0.99
                    target_params.data = target_weight * tao + (1 - tao) * online_weight
            for parameter in self.target_x_attn.parameters():
                parameter.requires_grad = False
        # online and target
        attn_feature_1 = self.online_x_attn(sub_feature_1, crop_feature)
        attn_feature_2 = self.target_x_attn(sub_feature_2, crop_feature)
        attn_feature_3 = self.online_x_attn(sub_feature_3, crop_feature)
        attn_feature_4 = self.target_x_attn(sub_feature_4, crop_feature)

        # loss
        loss_1 = loss_fn(attn_feature_1, attn_feature_2)
        loss_2 = loss_fn(attn_feature_3, attn_feature_4)
        loss = loss_1 + loss_2

        return loss.mean()
    
    
    # NS1 with 8 cubes for each branch
class SimAttention_8(nn.Module):
    def __init__(self,
                 aug_function,
                 sub_function,
                 cube_function,
                 crossed_attention_method):
        super().__init__()
        self.aug_function = aug_function
        self.sub_function = sub_function
        self.cube_function = cube_function
        
        self.online_encoder = online_encoder
        self.target_encoder = None

        self.online_x_attn = crossed_attention_method
        self.target_x_attn = None

    def forward(self, x):
        x = x.cpu().numpy()
        aug1, aug2 = self.aug_function(x), self.aug_function(x)

        # B, 1024, 3
        sub1 = torch.Tensor(self.sub_function(aug1, 1024)).to(device)
        sub2 = torch.Tensor(self.sub_function(aug2, 1024)).to(device)
        
        # B, 8, 1024, 3
        cube1 = torch.Tensor(self.cube_function(aug1, 8, 0.2, 1024)).to(device)
        cube2 = torch.Tensor(self.cube_function(aug2, 8, 0.2, 1024)).to(device)

        # [B, 1, N_f] N_f: output dimension of mlp: 512
        sub_feature_1 = self.online_encoder(sub1)
        sub_feature_3 = self.online_encoder(sub2)

        # with momentum encoder
        with torch.no_grad():
            if self.target_encoder is None:
                self.target_encoder = copy.deepcopy(self.online_encoder)
            else:
                for online_params, target_params in zip(self.online_encoder.parameters(),
                                                        self.target_encoder.parameters()):
                    target_weight, online_weight = target_params.data, online_params.data
                    # moving average decay is tao
                    tao = 0.99
                    target_params.data = target_weight * tao + (1 - tao) * online_weight
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad = False
            sub_feature_2 = self.target_encoder(sub2)
            sub_feature_4 = self.target_encoder(sub1)


        # cube feature  [B, 1, N_f]
        cube_feature_1_1 = self.online_encoder(cube1[:, 0, :, :].reshape(-1, 1024, 3).continuous())
        cube_feature_1_2 = self.online_encoder(cube1[:, 1, :, :].reshape(-1, 1024, 3).continuous())
        cube_feature_1_3 = self.online_encoder(cube1[:, 2, :, :].reshape(-1, 1024, 3).continuous())
        cube_feature_1_4 = self.online_encoder(cube1[:, 3, :, :].reshape(-1, 1024, 3).continuous())
        cube_feature_1_5 = self.online_encoder(cube1[:, 4, :, :].reshape(-1, 1024, 3).continuous())
        cube_feature_1_6 = self.online_encoder(cube1[:, 5, :, :].reshape(-1, 1024, 3).continuous())
        cube_feature_1_7 = self.online_encoder(cube1[:, 6, :, :].reshape(-1, 1024, 3).continuous())
        cube_feature_1_8 = self.online_encoder(cube1[:, 7, :, :].reshape(-1, 1024, 3).continuous())

        cube_feature_2_1 = self.online_encoder(cube2[:, 0, :, :].reshape(-1, 1024, 3).continuous())
        cube_feature_2_2 = self.online_encoder(cube2[:, 1, :, :].reshape(-1, 1024, 3).continuous())
        cube_feature_2_3 = self.online_encoder(cube2[:, 2, :, :].reshape(-1, 1024, 3).continuous())
        cube_feature_2_4 = self.online_encoder(cube2[:, 3, :, :].reshape(-1, 1024, 3).continuous())
        cube_feature_2_5 = self.online_encoder(cube2[:, 4, :, :].reshape(-1, 1024, 3).continuous())
        cube_feature_2_6 = self.online_encoder(cube2[:, 5, :, :].reshape(-1, 1024, 3).continuous())
        cube_feature_2_7 = self.online_encoder(cube2[:, 6, :, :].reshape(-1, 1024, 3).continuous())
        cube_feature_2_8 = self.online_encoder(cube2[:, 7, :, :].reshape(-1, 1024, 3).continuous())


        # crop feature concat [B, 8, N_f]
        crop_feature_1 = torch.cat((cube_feature_1_1, cube_feature_1_2,
                                    cube_feature_1_3, cube_feature_1_4,
                                    cube_feature_1_5, cube_feature_1_6,
                                    cube_feature_1_7, cube_feature_1_8,), dim=1)
        crop_feature_2 = torch.cat((cube_feature_2_1, cube_feature_2_2,
                                    cube_feature_2_3, cube_feature_2_4,
                                    cube_feature_2_5, cube_feature_2_6,
                                    cube_feature_2_7, cube_feature_2_8,), dim=1)
        # [B, 16, N_f]
        crop_feature = torch.cat((crop_feature_1, crop_feature_2), dim=1)

        # attention feature
        with torch.no_grad():
            if self.target_x_attn is None:
                self.target_x_attn = copy.deepcopy(self.online_x_attn)
            else:
                for online_params, target_params in zip(self.online_x_attn.parameters(),
                                                        self.target_x_attn.parameters()):
                    target_weight, online_weight = target_params.data, online_params.data
                    # moving average decay is tao
                    tao = 0.99
                    target_params.data = target_weight * tao + (1 - tao) * online_weight
            for parameter in self.target_x_attn.parameters():
                parameter.requires_grad = False
        
        # online and target
        attn_feature_1 = self.online_x_attn(sub_feature_1, crop_feature)
        attn_feature_2 = self.target_x_attn(sub_feature_2, crop_feature)
        attn_feature_3 = self.online_x_attn(sub_feature_3, crop_feature)
        attn_feature_4 = self.target_x_attn(sub_feature_4, crop_feature)

        # loss
        loss_1 = loss_fn(attn_feature_1, attn_feature_2)
        loss_2 = loss_fn(attn_feature_3, attn_feature_4)
        loss = loss_1 + loss_2

        return loss.mean()

