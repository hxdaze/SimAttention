import numpy as np
from cubes import fps
import torch
import random

# Slice Method
# patch（3 slices， 40% per slice）
def get_slice_index(index_slice, ratio_per_slice, overlap_ratio, num_all_points):
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
    return patch


def b_get_slice(point_set, xyz_dim, npoints=1024):
    """
    point_set: original point clouds
    xyz_dim: based on which axis to get slice
    npoints: each slice point number

    return: 2 random slices based on same axis
    """
    B, _, C = point_set.shape
    list_index_slice = random.sample(range(0, 3), 2)
    
    result_1 = numpy.ones((B, npoints, C))
    for i in range(B):
        result_1[i] = get_slice(point_set[i], xyz_dim, list_index_slice[0], npoints)
    
    result_2 = numpy.ones((B, npoints, C))
    for i in range(B):
        result_2[i] = get_slice(point_set[i], xyz_dim, list_index_slice[1], npoints)
    
    return torch.Tensor(result_1).cuda(), torch.Tensor(result_2).cuda()


