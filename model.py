import torch
import copy
import torch.nn as nn
import torch.nn.functional as F


# loss fn 损失函数
def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


# SimAttention Class
class SimAttention(nn.Module):
    def __init__(self,
                 sub1, sub2,
                 slice1, slice2,
                 cube1, cube2,
                 sphere1, sphere2,
                 o_encoder,
                 t_encoder,
                 attention_feature_method):
        super().__init__()
        self.sub1 = sub1
        self.sub2 = sub2
        self.slice1 = slice1
        self.slice2 = slice2
        self.cube1 = cube1
        self.cube2 = cube2
        self.sphere1 = sphere1
        self.sphere2 = sphere2
        self.online_encoder = o_encoder
        self.target_encoder = t_encoder

        # todo: can try later
        # self.mlp = projector_mlp

        self.attention_feature = attention_feature_method

    def forward(self, x):
        # [B, 1, N_f] N_f: output dimension of encoder
        sub_feature_1 = self.online_encoder(self.sub1)
        sub_feature_2 = self.target_encoder(self.sub2)

        # slice feature [B, 1, N_f]
        slice_feature_1 = self.online_encoder(self.slice1)
        slice_feature_2 = self.online_encoder(self.slice2)

        # cube feature  [B, 1, N_f]
        cube_feature_1 = self.online_encoder(self.cube1)
        cube_feature_2 = self.online_encoder(self.cube2)

        # sphere feature [B, 1, N_f]
        sphere_feature_1 = self.online_encoder(self.sphere1)
        sphere_feature_2 = self.online_encoder(self.sphere2)

        # crop feature concat [B, 3, N_f]
        crop_feature_1 = torch.cat((slice_feature_1, cube_feature_1, sphere_feature_1), dim=1)
        crop_feature_2 = torch.cat((slice_feature_2, cube_feature_2, sphere_feature_2), dim=1)
        # [B, 6, N_f]
        crop_feature = torch.cat((crop_feature_1, crop_feature_2), dim=1)

        # attention feature
        attn_feature_1 = self.attention_feature(sub_feature_1, crop_feature)
        attn_feature_2 = self.attention_feature(sub_feature_2, crop_feature)

        return loss_fn(attn_feature_1, attn_feature_2)


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
        x = x.numpy()
        aug1, aug2 = self.aug_function(x), self.aug_function(x)

        sub1, sub2 = torch.Tensor(self.sub_function(aug1, 1024)), torch.Tensor(
            self.sub_function(aug2, 1024))
        slice1, slice2 = torch.Tensor(self.slice_function(aug1, 1, 1, 1024)), torch.Tensor(
            self.slice_function(aug2, 1, 1, 1024))
        cube1, cube2 = torch.Tensor(self.cube_function(aug1, 0.2, 1024)), torch.Tensor(
            self.cube_function(aug2, 0.2, 1024))
        sphere1, sphere2 = torch.Tensor(self.sphere_function(aug1, 0.2, 1024)), torch.Tensor(
            self.sphere_function(aug2, 0.1, 1024))

        # [B, 1, N_f] N_f: output dimension of encoder
        sub_feature_1 = self.online_encoder(sub1)
        with torch.no_grad():
            self.target_encoder = copy.deepcopy(self.online_encoder)
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad = False
            sub_feature_2 = self.target_encoder(sub2)

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
        attn_feature_2 = self.attention_feature(sub_feature_2, crop_feature)

        return loss_fn(attn_feature_1, attn_feature_2)


# NS1
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
        x = x.numpy()
        aug1, aug2 = self.aug_function(x), self.aug_function(x)

        sub1, sub2 = torch.Tensor(self.sub_function(aug1, 1024)), torch.Tensor(
            self.sub_function(aug2, 1024))
        slice1, slice2 = torch.Tensor(self.slice_function(aug1, 1, 1, 1024)), torch.Tensor(
            self.slice_function(aug2, 1, 1, 1024))
        cube1, cube2 = torch.Tensor(self.cube_function(aug1, 0.2, 1024)), torch.Tensor(
            self.cube_function(aug2, 0.2, 1024))
        sphere1, sphere2 = torch.Tensor(self.sphere_function(aug1, 0.2, 1024)), torch.Tensor(
            self.sphere_function(aug2, 0.1, 1024))

        # [B, 1, N_f] N_f: output dimension of encoder
        sub_feature_1 = self.online_encoder(sub1)
        sub_feature_3 = self.online_encoder(sub2)
        # without momentum encoder
#         with torch.no_grad():
#             self.target_encoder = copy.deepcopy(self.online_encoder)
#             for parameter in self.target_encoder.parameters():
#                 parameter.requires_grad = False
#             sub_feature_2 = self.target_encoder(sub2)
#             sub_feature_4 = self.target_encoder(sub1)
        
        # with momentum encoder
        with torch.no_grad():
            if self.target_encoder == None:
                self.target_encoder = copy.deepcopy(self.online_encoder)
            else:
                for online_params, target_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
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
        attn_feature_2 = self.attention_feature(sub_feature_2, crop_feature)
        attn_feature_3 = self.attention_feature(sub_feature_3, crop_feature)
        attn_feature_4 = self.attention_feature(sub_feature_4, crop_feature)

        # loss
        loss_1 = loss_fn(attn_feature_1, attn_feature_2)
        loss_2 = loss_fn(attn_feature_3, attn_feature_4)
        loss = loss_1 + loss_2

        return loss


# NS2
class SimAttention_4(nn.Module):
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
        x = x.numpy()
        aug1, aug2 = self.aug_function(x), self.aug_function(x)

        sub1, sub2 = torch.Tensor(self.sub_function(aug1, 1024)), torch.Tensor(
            self.sub_function(aug2, 1024))
        slice1, slice2 = torch.Tensor(self.slice_function(aug1, 1, 1, 1024)), torch.Tensor(
            self.slice_function(aug2, 1, 1, 1024))
        cube1, cube2 = torch.Tensor(self.cube_function(aug1, 0.2, 1024)), torch.Tensor(
            self.cube_function(aug2, 0.2, 1024))
        sphere1, sphere2 = torch.Tensor(self.sphere_function(aug1, 0.2, 1024)), torch.Tensor(
            self.sphere_function(aug2, 0.1, 1024))

        # [B, 1, N_f] N_f: output dimension of encoder
        sub_feature_1 = self.online_encoder(sub1)
        sub_feature_3 = self.online_encoder(sub2)
        with torch.no_grad():
            self.target_encoder = copy.deepcopy(self.online_encoder)
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
        loss_1 = loss_fn(attn_feature_1, sub_feature_2)
        loss_2 = loss_fn(attn_feature_3, sub_feature_4)
        loss = loss_1 + loss_2

        return loss


# NS3
class SimAttention_5(nn.Module):
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
        x = x.numpy()
        aug1, aug2 = self.aug_function(x), self.aug_function(x)

        sub1, sub2 = torch.Tensor(self.sub_function(aug1, 1024)), torch.Tensor(
            self.sub_function(aug2, 1024))
        slice1, slice2 = torch.Tensor(self.slice_function(aug1, 1, 1, 1024)), torch.Tensor(
            self.slice_function(aug2, 1, 1, 1024))
        cube1, cube2 = torch.Tensor(self.cube_function(aug1, 0.2, 1024)), torch.Tensor(
            self.cube_function(aug2, 0.2, 1024))
        sphere1, sphere2 = torch.Tensor(self.sphere_function(aug1, 0.2, 1024)), torch.Tensor(
            self.sphere_function(aug2, 0.1, 1024))

        # [B, 1, N_f] N_f: output dimension of encoder
        sub_feature_1 = self.online_encoder(sub1)
        sub_feature_3 = self.online_encoder(sub2)
        with torch.no_grad():
            self.target_encoder = copy.deepcopy(self.online_encoder)
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

        # attention feature [B, 1, N_f]
        attn_feature_1 = self.attention_feature(sub_feature_1, crop_feature_2)
        attn_feature_2 = self.attention_feature(sub_feature_2, crop_feature_1)

        attn_feature_3 = self.attention_feature(sub_feature_3, crop_feature_2)
        attn_feature_4 = self.attention_feature(sub_feature_4, crop_feature_1)

        # loss
        loss_1 = loss_fn(attn_feature_1, attn_feature_2)
        loss_2 = loss_fn(attn_feature_3, attn_feature_4)
        loss = loss_1 + loss_2

        return loss
