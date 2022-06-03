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


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    device = batch_pc.device
    # batch_pc: BxNx3
    for b in range(batch_pc.shape[0]):
        dropout_ratio = np.random.random()*max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]
        drop_idx = torch.from_numpy(drop_idx).to(device)
        r = batch_pc.clone()
        if len(drop_idx) > 0:
            r[b, drop_idx, :] = batch_pc[b, 0, :]  # set to the first point
    return r


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    device = batch_data.device
    scales = np.random.uniform(scale_low, scale_high, B)
    scales = torch.from_numpy(scales).to(device)
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
    device = batch_data.device
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
    shifts = torch.from_numpy(shifts).to(device)
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc
  
def b_fps(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        result: sampled pointcloud , [B, npoint, 3]
    """

    device = xyz.device
    if xyz.dim() < 3:
        xyz = xyz.reshape((1, -1, 3))
    B, N, C = xyz.shape
    result = torch.zeros(B, npoint, C).to(device)

    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10

    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    barycenter = torch.sum((xyz), 1)
    barycenter = barycenter / xyz.shape[1]
    barycenter = barycenter.view(B, 1, 3)

    dist = torch.sum((xyz - barycenter) ** 2, -1, dtype=torch.float32)
    farthest = torch.max(dist, 1)[1]

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1, dtype=torch.float32)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    for b in range(0, B):
        for index in range(0, npoint):

            result[b, index] = xyz[b, centroids[b, index]]

    # r = [xyz[b, centroids[b, i]].item() for b in range(0, B) for i in range(0, npoint)]

    return result


# Slice Method
def b_get_slice(point_set, xyz_dim, index_slice, npoints):
    device = point_set.device
    B, _, C = point_set.shape
    result = torch.zeros((B, npoints, C)).to(device)

    def get_slice_index(index_slice, ratio_per_slice, overlap_ratio, num_all_points):
        start_index = index_slice * (ratio_per_slice - overlap_ratio) * num_all_points
        end_index = start_index + ratio_per_slice * num_all_points
        return int(start_index), int(end_index)

    def get_1_slice(point_set, xyz_dim, index_slice, npoints):
        # xyz_dim: 0, 1, 2 for x, y, z
        start_index, end_index = get_slice_index(index_slice, 0.4, 0.1, len(point_set))
        patch_index = torch.argsort(point_set, dim=0)[start_index: end_index, xyz_dim]
        patch = point_set[patch_index]

        if len(patch_index) > npoints:
            patch = b_fps(patch, npoints)
        return patch

    for b in range(0, B):
        result[b] = get_1_slice(point_set[b], xyz_dim, index_slice, npoints)
    return result


# Cube Method
def b_get_cube(point_set, side_length, npoints):
    device = point_set.device
    B, _, C = point_set.shape
    result = torch.zeros((B, npoints, C)).to(device)

    def point_in_cube(point_xyz, side_length):
        flag = True
        for i in range(0, len(point_xyz)):
            if abs(point_xyz[i]) >= (side_length / 2):
                flag = False
                break
        return flag

    def get_1_cube(point_set, side_length, npoints):
        sample_index = []
        for i in range(0, len(point_set)):
            if point_in_cube(point_set[i], side_length):
                sample_index.append(i)
        if len(sample_index) >= npoints:
            r = b_fps(point_set[sample_index], npoints)
            return r
        else:
            return get_1_cube(point_set, side_length + 0.1, npoints)

    for i in range(point_set.shape[0]):
        result[i] = get_1_cube(point_set[i], side_length, npoints)
    return result


# Sphere Method
def b_get_sphere(point_set, radius, npoints):
    device = point_set.device
    B, _, C = point_set.shape
    result = torch.zeros((B, npoints, C)).to(device)

    def point_in_ball(point_xyz, center_xyz, radius):
        flag = False
        dist = 0
        for i in range(3):
            dist += (point_xyz[i] - center_xyz[i]) ** 2
        if dist <= radius ** 2:
            flag = True
        return flag

    def get_1_sphere(point_set, radius, npoints):
        center_xyz = torch.zeros([3]).to(device)
        sample_index = []
        for i in range(0, len(point_set)):
            if point_in_ball(point_set[i], center_xyz, radius):
                sample_index.append(i)
        if len(sample_index) >= npoints:
            r = b_fps(point_set[sample_index], npoints)
            return r
        else:
            return get_1_sphere(point_set, radius + 0.1, npoints)

    for i in range(point_set.shape[0]):
        result[i] = get_1_sphere(point_set[i], radius, npoints)
    return result
