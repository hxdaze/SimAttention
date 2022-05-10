import copy
import torch
import numpy as np
"""
utilization functions
"""


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def get_target_encoder(online_encoder):
    target_encoder = copy.deepcopy(online_encoder)
    set_requires_grad(target_encoder, False)
    return target_encoder


def get_attention_feature(q, kv):
    """
    - function: get attention features
    - input: 1 feature from subsample branch, 1 concat feature
    - output: crossed attention
    - parameters:
    - data dimension: ([B, 1, new_F], [B, 6, new_F]) ---> [B, 1, new_F]
    """
    kv_t = kv.permute(0, 2, 1)
    energy = torch.bmm(q, kv_t)
    a = torch.bmm(energy, kv)
    return a.reshape(q.shape[0], -1)


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    # batch_pc: BxNx3
    for b in range(batch_pc.shape[0]):
        dropout_ratio = np.random.random()*max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]  # set to the first point
    return batch_pc


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data


def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data