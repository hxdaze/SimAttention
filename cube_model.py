import torch
import copy
import torch.nn as nn

from model import CrossedAttention, loss_fn


class SimAttention_All_Cubes(nn.Module):
    """
    crop method uses all cubes 
    """
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
        sub1 = torch.Tensor(self.sub_function(aug1, 1024)).cuda()
        sub2 = torch.Tensor(self.sub_function(aug2, 1024)).cuda()
        
        # B, 8, 1024, 3
        cube1 = torch.Tensor(self.cube_function(aug1, 8, 0.2, 1024)).cuda()
        cube2 = torch.Tensor(self.cube_function(aug2, 8, 0.2, 1024)).cuda()

        # [B, 1, N_f] N_f: output dimension of mlp: 1024
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

        # momentum attention feature
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