import numpy as np
import open3d as o3d
import os
import progressbar
import scipy.signal as signal
import shelve
import time

from matplotlib import pyplot as plt
from tkinter import filedialog

# Find floor height
# Find beam height
# Check if they're the same (based on expected density)
# Detect major lines
# Compare to walls
# Make basic material judgement
# Decide Splits
# Detect columns


# === DEFINITIONS ===

aabb_main = None

color_back = 'gray'
color_back_highlight = 'cyan'
color_front = 'C0'
color_front_highlight = 'cyan'


def set_up_vector(vis):
    vis.get_view_control().set_up((0.001, 0.001, 0.9999))


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


def smooth_histogram(hist, extension=1):
    """ugly for now but meh"""
    hist_copy = np.copy(hist)
    for i in range(len(hist)):
        hist[i] = 0
        for j in range(-extension,extension + 1):
            new_i = i + j
            if new_i < 0 or new_i >= len(hist):
                continue
            hist[i] += hist_copy[new_i]
    return hist


def normalize_histogram(hist):
    hist = hist.astype('float')
    hist /= np.max(hist)
    return hist


def process_histogram(hist):
    hist_smooth = smooth_histogram(np.copy(hist))
    hist_smooth = normalize_histogram(hist_smooth)
    hist = normalize_histogram(hist)
    return hist, hist_smooth


def analyze_z_levels(pc, aabb):
    points = np.asarray(pc.points)

    bins = 256

    hist_z, bin_edges = np.histogram(points[:, 2], bins)
    hist_z, hist_z_smooth = process_histogram(hist_z)
    print(np.var(hist_z))

    peaks, properties = signal.find_peaks(hist_z_smooth, width=3, prominence=0.1)

    fig, axs = plt.subplots(3,3)
    bar_list_z_smooth = axs[0, 0].bar(range(len(hist_z_smooth)), hist_z_smooth,color=color_back)
    bar_list_z = axs[0, 0].bar(range(len(hist_z)), hist_z, color=color_front)

    axs[0, 1].axis('off')
    axs[0, 2].axis('off')
    plt.setp(axs[0, 0], ylabel='Z Axis')
    plt.setp(axs[1, 0], ylabel='X Axis')
    plt.setp(axs[2, 0], ylabel='Y Axis')

    for i, peak in enumerate(peaks):
        bar_list_z_smooth[peak].set_color(color_back_highlight)
        bar_list_z[peak].set_color(color_front_highlight)

        pc_slice = get_slice(pc, aabb, 2, peak / bins, 3 / bins, normalized=True)
        slice_points = np.asarray(pc_slice.points)

        hist_x, bins_x = np.histogram(slice_points[:, 0], bins)
        hist_x, hist_x_smooth = process_histogram(hist_x)
        peaks_x, _ = signal.find_peaks(hist_x_smooth, width=3, prominence=0.1)

        hist_y, bins_y = np.histogram(slice_points[:, 1], bins)
        hist_y, hist_y_smooth = process_histogram(hist_y)
        peaks_y, _ = signal.find_peaks(hist_y_smooth, width=3, prominence=0.1)

        bar_list_x_smooth = axs[1, i].bar(range(len(hist_x_smooth)), hist_x_smooth, color=color_back)
        bar_list_x = axs[1, i].bar(range(len(hist_x)), hist_x)

        bar_list_y_smooth = axs[2, i].bar(range(len(hist_y_smooth)), hist_y_smooth, color=color_back)
        bar_list_y = axs[2, i].bar(range(len(hist_y)), hist_y)

        for peak_x in peaks_x:
            bar_list_x_smooth[peak_x].set_color(color_back_highlight)
            bar_list_x[peak_x].set_color(color_front_highlight)
        for peak_y in peaks_y:
            bar_list_y_smooth[peak_y].set_color(color_back_highlight)
            bar_list_y[peak_y].set_color(color_front_highlight)
        print(peak)
        print(np.var(hist_x))
        print(np.var(hist_y))

    plt.show()



def setup_vis():
    vis = o3d.visualization.VisualizerWithKeyCallback()
    #vis.register_key_callback(83, save_view)
    vis.register_key_callback(69, set_up_vector)
    vis.create_window()
    vis.get_render_option().point_size = 1

    return vis


# === MAIN SCRIPT ===


if __name__ == '__main__':
    filepath = filedialog.askopenfilename(initialdir=os.getcwd(), title="Choose Input Cloud")
    dirname = os.path.dirname(filepath) + "/"
    basename = os.path.basename(filepath)
    filename = os.path.splitext(basename)[0]

    project_data = shelve.open(dirname + filename)
    project_data["test"] = 2
    project_data.close()

    vis = setup_vis()

    cloud = o3d.io.read_point_cloud(filepath)
    vis.add_geometry(cloud)

    aabb_main = cloud.get_axis_aligned_bounding_box()
    aabb_main.color = (1, 0, 0)
    vis.add_geometry(aabb_main)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000, origin=[0, 0, 0])
    vis.add_geometry(mesh_frame)

    analyze_z_levels(cloud, aabb_main)

    #print(peaks)
    #print(properties)

    vis.run()
    vis.destroy_window()
