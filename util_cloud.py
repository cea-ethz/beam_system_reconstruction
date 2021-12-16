import numpy as np
import open3d as o3d
import random


def get_slice(pc, aabb, axis, position, width, normalized=False):
    min_main = aabb.get_min_bound()
    max_main = aabb.get_max_bound()

    bb_range = max_main - min_main

    if normalized:
        position = (bb_range[axis] * position) + min_main[axis]
        width = bb_range[axis] * width

    new_min = np.copy(min_main)
    new_max = np.copy(max_main)

    new_min[axis] = position - (width / 2)
    new_max[axis] = position + (width / 2)

    bb = o3d.geometry.AxisAlignedBoundingBox(new_min, new_max)
    pc_slice = pc.crop(bb)
    return pc_slice


def split_slice(pc, aabb, axis, position, width, normalized=False):
    """Return slice as well as region outside slice as a separate cloud"""
    min_main = aabb.get_min_bound()
    max_main = aabb.get_max_bound()

    bb_range = max_main - min_main

    if normalized:
        position = (bb_range[axis] * position) + min_main[axis]
        width = bb_range[axis] * width

    min_a = np.copy(min_main)
    max_a = np.copy(max_main)

    min_b = np.copy(min_main)
    max_b = np.copy(max_main)

    min_c = np.copy(min_main)
    max_c = np.copy(max_main)

    max_a[axis] = position - (width / 2)

    min_b[axis] = position - (width / 2)
    max_b[axis] = position + (width / 2)

    min_c[axis] = position + (width / 2)

    bb_a = o3d.geometry.AxisAlignedBoundingBox(min_a, max_a)
    bb_b = o3d.geometry.AxisAlignedBoundingBox(min_b, max_b)
    bb_c = o3d.geometry.AxisAlignedBoundingBox(min_c, max_c)

    slice_a = pc.crop(bb_a)
    slice_b = pc.crop(bb_b)
    slice_c = pc.crop(bb_c)

    slice_a += slice_c

    return slice_b, slice_a


def flatten_cloud(pc):
    """Project a cloud to the xy plane"""
    points = np.asarray(pc.points)
    points[:, 2] = 0
    pc.points = o3d.utility.Vector3dVector(points)
    return pc


def flatten_to_axis(point_array, axis):
    assert 0 <= axis <= 2
    ip_new = np.zeros((len(point_array), 2))

    if axis == 0:
        ip_new[:, 0] = point_array[:, 1]
        ip_new[:, 1] = point_array[:, 2]
    elif axis == 1:
        ip_new[:, 0] = point_array[:, 0]
        ip_new[:, 1] = point_array[:, 2]
    elif axis == 2:
        ip_new[:, 0] = point_array[:, 0]
        ip_new[:, 1] = point_array[:, 1]

    return ip_new


def split_by_labels(pc, labels, salt_z_axis=True):
    """
    Returns a new cloud for each unique label.

    :param pc: Input point cloud
    :param labels: Array containing label id for each point in input cloud
    :param salt_z_axis: Used when splitting flattened clouds : sets the last point's z-value to 0.01, to make bounding boxes work
    :return: Array of point clouds
    """

    points = np.asarray(pc.points)
    colors = np.asarray(pc.colors)

    # Drop label '-1', representing unlabeled points
    labelset, label_counts = np.unique(labels, return_counts=True)

    output = []

    for label, count in zip(labelset, label_counts):
        if label == -1 or count < 100:
            continue

        inclusion = labels == label
        sub_points = points[inclusion]
        sub_colors = colors[inclusion]

        if salt_z_axis:
            #sub_points[-1, 2] = 0.001
            sub_points[-0, 2] = -5
            sub_points[-1, 2] = 5

        cloud = o3d.geometry.PointCloud()

        cloud.points = o3d.utility.Vector3dVector(sub_points)
        cloud.colors = o3d.utility.Vector3dVector(sub_colors)

        output.append(cloud)

    return output


def check_aabb_overlap_2d(a, b):
    center_a = a.get_center()
    center_b = b.get_center()

    half_a = a.get_half_extent()
    half_b = b.get_half_extent()

    return abs(center_a[0] - center_b[0]) < half_a[0] + half_b[0] and abs(center_a[1] - center_b[1]) < half_a[1] + \
           half_b[1]


def chamfer_distance(a, b):
    """
    Returns the chamfer distance between two point clouds.
    Assumes they've been pre-aligned/registered

    :param a:
    :param b:
    :return:
    """
    #print("Chamfer Distance with Point Counts : {}, {}".format(len(a.points), len(b.points)))

    dist_a = np.asarray(a.compute_point_cloud_distance(b))
    dist_b = np.asarray(b.compute_point_cloud_distance(a))

    dist_a = np.sum(dist_a) / len(dist_a)
    dist_b = np.sum(dist_b) / len(dist_b)

    return dist_a + dist_b


def cloud_to_accumulator(points, scale=8):
    """
    Turns an internal point array from a point cloud into an accumulator image. Assumes points are already XY plane oriented

    :param points: Numpy array of points
    :param scale: Cloud downscaling - i.e. number of millimeters per pixel
    :return: Grayscale image of range 0-255
    """

    min_x = int(np.min(points[:, 0]))
    min_y = int(np.min(points[:, 1]))
    max_x = int(np.max(points[:, 0]))
    max_y = int(np.max(points[:, 1]))
    range_x = max_x - min_x
    range_y = max_y - min_y

    accumulator = np.zeros((range_x // scale, range_y // scale))

    for point in points:
        x = int((point[0] - min_x) // scale)
        y = int((point[1] - min_y) // scale)

        accumulator[x-5:x+5, y-5:y+5] += 1

    accumulator /= np.max(accumulator)
    accumulator = np.float32(accumulator)
    accumulator *= 255

    return accumulator