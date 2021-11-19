import numpy as np
import open3d as o3d

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