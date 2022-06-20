import random
import torch
import numpy
import numpy as np

"""
@author: HQL
function 1: fps
function 2: get_slice
function 3: get_cube
function 4: get_sphere
function b: with batch data
"""

# FPS Method
# 需要计算重心坐标的算法
def fps(original, npoints):
    center_xyz = np.sum(original, 0)
    # 得到重心点的坐标
    center_xyz = center_xyz / len(original)
    # 计算出初始的最远点
    dist = np.sum((original - center_xyz) ** 2, 1)
    farthest = np.argmax(dist)
    distance = np.ones(len(original)) * 1e10
    target_index = np.zeros(npoints, dtype=np.int32)

    for i in range(npoints):
        target_index[i] = farthest
        target_point_xyz = original[target_index[i], :]

        dist = np.sum((original - target_point_xyz) ** 2, 1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)

    return original[target_index]


def b_fps(original, npoints):
    result = []
    for i in range(original.shape[0]):
        result.append(fps(original[i], npoints))
    return np.array(result)


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def b_FPS(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids, index_points(xyz, centroids)


def k_points(a, b, k):
    # a: small, b: big one
    inner = -2 * torch.matmul(a, b.transpose(2, 1))
    aa = torch.sum(a**2, dim=2, keepdim=True)
    bb = torch.sum(b**2, dim=2, keepdim=True)
    pairwise_distance = -aa - inner - bb.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def new_k_patch(x, k=2048, n_patch=8, n_points=1024):
    patch_centers_index, _ = b_FPS(x, n_patch)  # torch.Size([B, n_patch])
    center_point_xyz = index_points(x, patch_centers_index)  # [B, n_patch]

    idx = k_points(center_point_xyz, x, k)  # B, k, 2048
    idx = idx.permute(0, 2, 1)  # B, k, n_patch

    new_patch = torch.zeros([n_patch, x.shape[0], n_points, x.shape[-1]]).to(device)
    for i in range(n_patch):
        patch_idx = idx[:, :, i].reshape(x.shape[0], -1)
        _, patch_points = b_FPS(index_points(x, patch_idx), n_points)
        new_patch[i] = patch_points
    new_patch = new_patch.permute(1, 0, 2, 3)   # torch.Size([4, 8, 1024, 3])
    return new_patch


# Slice Method
# patch（3 slices， 40% per slice）
def get_slice_index(index_slice, ratio_per_slice, overlap_ratio, num_all_points):
    # todo：等会儿就把这个写死就行
    # index_slice = 0, 1, 2
    start_index = index_slice * (ratio_per_slice - overlap_ratio) * num_all_points
    end_index = start_index + ratio_per_slice * num_all_points
    return int(start_index), int(end_index)


def get_slice(point_set, xyz_dim, index_slice, npoints):
    # xyz_dim: 0, 1, 2 for x, y, z
    start_index, end_index = get_slice_index(index_slice, 0.4, 0.1, len(point_set))
    patch_index = np.argsort(point_set, axis=0)[start_index: end_index, xyz_dim]
    patch = point_set[patch_index]
    random.shuffle(patch)
    if len(patch_index) > npoints:
        patch = fps(patch, npoints)
    # 返回slice后的值，按照哪个维度切的 xyz，index_slice也就是第几个
    # return patch, xyz_dim, index_slice
    return patch


def b_get_slice(point_set, xyz_dim, index_slice, npoints):
    B, _, C = point_set.shape
    result = numpy.ones((B, npoints, C))
    for i in range(point_set.shape[0]):
        result[i] = get_slice(point_set[i], xyz_dim, index_slice, npoints)
    return result


# Cube Method
# 判断一个点是否在cube内
def point_in_cube(point_xyz, side_length):
    # point_xyz是点的坐标
    # side_length是cube的边长
    flag = True
    for i in range(0, len(point_xyz)):
        if abs(point_xyz[i]) >= (side_length / 2):
            flag = False
            break
    return flag


def get_cube(point_set, side_length, npoints):
    output_samples = []
    for i in range(0, len(point_set)):
        if point_in_cube(point_set[i], side_length):
            output_samples.append(i)
    samples = point_set[output_samples]
    if len(output_samples) >= npoints:
        result = fps(samples, npoints)
        # return samples, result
        return result
    else:
        return get_cube(point_set, side_length + 0.1, npoints)


def b_get_cube(point_set, side_length, npoints):
    B, _, C = point_set.shape
    result = numpy.ones((B, npoints, C))
    for i in range(point_set.shape[0]):
        result[i] = get_cube(point_set[i], side_length, npoints)
    return result


# Sphere Method
# 判断一个点是否在球内
def point_in_ball(point_xyz, center_xyz, radius):
    # point_xyz和center_xyz 分别是点和球心的坐标
    # radius是球的半径
    flag = False
    dist = 0
    for i in range(3):
        dist += (point_xyz[i] - center_xyz[i]) ** 2
    if dist <= radius ** 2:
        flag = True
    return flag


def get_sphere(point_set, radius, npoints):
    center_xyz = np.zeros([3], dtype=np.float32)
    output_samples = []
    for i in range(0, len(point_set)):
        if point_in_ball(point_set[i], center_xyz, radius):
            output_samples.append(i)
    samples = point_set[output_samples]
    if len(output_samples) >= npoints:
        result = fps(samples, npoints)
        # return samples, result
        return result
    else:
        return get_sphere(point_set, radius + 0.1, npoints)


def b_get_sphere(point_set, radius, npoints):
    B, _, C = point_set.shape
    result = numpy.ones((B, npoints, C))
    for i in range(point_set.shape[0]):
        result[i] = get_sphere(point_set[i], radius, npoints)
    return result
