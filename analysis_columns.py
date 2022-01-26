import numpy as np
import open3d as o3d

from matplotlib import pyplot as plt

import ui
import settings
import util_cloud

from BIM_Geometry import Column


# TODO :  Draw and export column images for debug / documentation
def analyze_columns(pc, aabb, pc_main, aabb_main, primary_beams, z_extents):
    pc_flat = util_cloud.flatten_cloud(pc)
    eps = settings.read("tuning.dbscan_eps")
    #labels = np.array(pc_flat.cluster_dbscan(eps=12, min_points=10))
    labels = np.array(pc_flat.cluster_dbscan(eps=eps, min_points=10))

    # If column detection failed
    if len(labels) == 0:
        return []

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pc_flat.colors = o3d.utility.Vector3dVector(colors[:, :3])

    subclouds = util_cloud.split_by_labels(pc_flat, labels)

    output_columns = []

    for subcloud in subclouds:
        aabb_subcloud = subcloud.get_axis_aligned_bounding_box()
        extent = aabb_subcloud.get_extent()
        extent_max = max(extent[0], extent[1])
        extent_min = min(extent[0], extent[1])

        if settings.read("visibility.column_clusters"):
            ui.vis.add_geometry(subcloud)
            ui.vis.add_geometry(aabb_subcloud)

        # Test candidates for correct dimensions
        if extent_min != 0.0 and \
           100 <= extent[0] <= 500 and \
           100 <= extent[1] <= 500 and \
           extent_max / extent_min < 2:
            # Ensure that candidates touch at least one primary layer beam
            for beam in primary_beams:
                if util_cloud.check_aabb_overlap_2d(aabb_subcloud, beam.aabb):
                    column = Column(aabb_subcloud.get_center())
                    column.child_beams.append(beam)
                    output_columns.append(column)
                    break

    # Calculate final cloud and aabb
    for column in output_columns:
        crop_min = np.copy(column.center)
        crop_max = np.copy(column.center)

        crop_min[0] -= 100
        crop_min[1] -= 100
        crop_min[2] = z_extents[0]

        crop_max[0] += 100
        crop_max[1] += 100
        crop_max[2] = z_extents[1]
        crop = o3d.geometry.AxisAlignedBoundingBox(crop_min, crop_max)

        column.pc = pc_main.crop(crop)

        column.pc = util_cloud.filter_std(column.pc, 0, factor=1.5)
        column.pc = util_cloud.filter_std(column.pc, 1, factor=1.5)

        column.aabb = column.pc.get_axis_aligned_bounding_box()

    return output_columns





