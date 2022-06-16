import numpy as np


def fps(original, npoints=1024):
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


def get_random_center():
    u = np.random.uniform(-1.0, 1.0)
    theta = 2 * np.pi * np.random.uniform(0.0, 2)
    
    x = np.power((1 - u * u), 1/2) * np.cos(theta)
    x = np.abs(x)
    x = np.random.uniform(-x, x)
    y = np.power((1 - u * u), 1/2) * np.sin(theta)
    y = np.abs(y)
    y = np.random.uniform(-y, y)
    z = u
    return (x, y, z)


def point_in_cube(point_xyz, center_xyz, side_length):
    flag = True
    for i in range(0, len(point_xyz)):
        if abs(point_xyz[i] - center_xyz[i]) >= (side_length / 2):
            flag = False
            break
    return flag


def get_1_cube(point_set, center_xyz, side_length, npoints=1024):
    # 现在需要从单个点云中找到8个cube出来
    # 存储满足条件的点的索引
    output_samples = []
    for i in range(0, len(point_set)):
        if point_in_cube(point_set[i], center_xyz, side_length):
            output_samples.append(i)
    samples = point_set[output_samples]
    if len(output_samples) >= npoints:
        result = fps(samples, npoints)
        return result
    else:
        return get_1_cube(point_set, center_xyz, side_length + 0.2, npoints)

    
def get_cubes(point_set, num_cube=8, side_length, npoints=1024):
    # 保存结果
    result = np.ones((num_cube, npoints, 3))
    for i in range(0, num_cube):
        # 随机生成一个cube中心
        center = get_random_center()
        result[i] = get_1_cube(point_set, center, side_length, npoints)
    return result


def b_get_cubes(b_point_set, num_cube=8, side_length, npoints=1024):
    B = b_point_set.shape[0]
    result = numpy.ones((B, num_cube, npoints, 3))
    for i in range(B):
        result[i] = get_cubes(b_point_set[i], num_cube, side_length, npoints)
    return result
