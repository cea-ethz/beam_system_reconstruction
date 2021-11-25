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


def split_by_labels(pc, labels, salt_z_axis = True):
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
            sub_points[-1,2] = 0.001

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

    return abs(center_a[0] - center_b[0]) < half_a[0] + half_b[0] and abs(center_a[1] - center_b[1]) < half_a[1] + half_b[1]




