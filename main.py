import numpy as np
import open3d as o3d
import os
import progressbar
import scipy.signal as signal
import shelve
import time

from matplotlib import pyplot as plt
from tkinter import filedialog


# Adjust lengths of major lines
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

variance_split = 0.075

bins = 256

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
        for j in range(-extension, extension + 1):
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


def get_peak_slice_params(hist, peak, diff=0.1):
    """
    Given a peak, return the slice parameters to stay within [diff] of that peak (may not be centered at the peak)
    :param hist:
    :param peak:
    :param diff:
    :return:
    """
    low = peak
    high = peak

    for i in range(low - 1,-1,-1):
        if hist[peak] - hist[i] < diff:
            low = i
        else:
            break
    for i in range(low + 1, len(hist)):
        if hist[peak] - hist[i] < diff:
            high = i
        else:
            break

    width = high - low
    position = width / 2 + low
    return position, width


def analyze_z_levels(pc, aabb):
    points = np.asarray(pc.points)

    hist_z, bin_edges = np.histogram(points[:, 2], bins)
    hist_z, hist_z_smooth = process_histogram(hist_z)
    mean_z = np.mean(hist_z_smooth)

    peaks, properties = signal.find_peaks(hist_z_smooth, width=3, prominence=0.1)

    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    fig, axs = plt.subplots(3, 3, figsize=(1920*px, 1080*px))
    bar_list_z_smooth = axs[0, 0].bar(range(len(hist_z_smooth)), hist_z_smooth, color=color_back, width=1)
    bar_list_z = axs[0, 0].bar(range(len(hist_z)), hist_z, color=color_front, width=1)
    axs[0, 0].axhline(mean_z, color='orange')

    axs[0, 1].axis('off')
    #axs[0, 2].axis('off')
    plt.setp(axs[0, 0], ylabel='Z Axis')
    plt.setp(axs[1, 0], ylabel='X Axis')
    plt.setp(axs[2, 0], ylabel='Y Axis')

    for i, peak in enumerate(peaks):
        # Highlight peaks in Z-plot
        bar_list_z_smooth[peak].set_color(color_back_highlight)
        bar_list_z[peak].set_color(color_front_highlight)

        # Get extents of peak
        peak_slice_position, peak_slice_width = get_peak_slice_params(hist_z_smooth, peak, 0.1)

        # Get slice at Z height
        pc_slice = get_slice(pc, aabb, 2, peak_slice_position / bins, peak_slice_width / bins, normalized=True)
        pc_slice_aabb = pc_slice.get_axis_aligned_bounding_box()
        slice_points = np.asarray(pc_slice.points)

        rel_height = 0.75 # Check the width near the bottom of the peak
        prominence = 0.13 # Experimentally tuned, this should be determined more exactly
        # Take histogram along X axis
        hist_x, bins_x = np.histogram(slice_points[:, 0], bins)
        hist_x, hist_x_smooth = process_histogram(hist_x)
        mean_x = np.mean(hist_x_smooth)
        peaks_x, _ = signal.find_peaks(hist_x_smooth, width=4, prominence=prominence, rel_height=rel_height)

        # Take histogram along Y Axis
        hist_y, bins_y = np.histogram(slice_points[:, 1], bins)
        hist_y, hist_y_smooth = process_histogram(hist_y)
        mean_y = np.mean(hist_y_smooth)
        peaks_y, _ = signal.find_peaks(hist_y_smooth, width=4, prominence=prominence, rel_height=rel_height)

        # Plot X axis histogram and mean
        bar_list_x_smooth = axs[1, i].bar(range(len(hist_x_smooth)), hist_x_smooth, color=color_back, width=1)
        bar_list_x = axs[1, i].bar(range(len(hist_x)), hist_x, width=1)
        axs[1, i].axhline(mean_x, color='orange')

        # Plot Y axis histogram and mean
        bar_list_y_smooth = axs[2, i].bar(range(len(hist_y_smooth)), hist_y_smooth, color=color_back, width=1)
        bar_list_y = axs[2, i].bar(range(len(hist_y)), hist_y, width=1)
        axs[2, i].axhline(mean_y, color='orange')

        # Highlight peaks in X and Y histograms
        for peak_x in peaks_x:
            bar_list_x_smooth[peak_x].set_color(color_back_highlight)
            bar_list_x[peak_x].set_color(color_front_highlight)
        for peak_y in peaks_y:
            bar_list_y_smooth[peak_y].set_color(color_back_highlight)
            bar_list_y[peak_y].set_color(color_front_highlight)

        # Calculate variance on each axis
        variance_x = np.var(hist_x)
        variance_y = np.var(hist_y)

        if variance_x < variance_split or variance_y < variance_split:
            o3d.io.write_point_cloud(dir_output + filename + "_grid_{}.ply".format(peak), pc_slice)
            analyze_beam_system(pc_slice, pc_slice_aabb, 0, hist_x_smooth, peaks_x)
            analyze_beam_system(pc_slice, pc_slice_aabb, 1, hist_y_smooth, peaks_y)

    plt.savefig(dir_output + filename + "_plot.png")
    plt.show()


def analyze_beam_system(pc, aabb, axis, hist, peaks):
    global dumb_flag

    for peak in peaks:
        slice_position, slice_width = get_peak_slice_params(hist, peak, 0.1)
        beam_slice = get_slice(pc, aabb, axis, slice_position / bins, slice_width / bins, normalized=True)
        beam_slice_points = np.array(beam_slice.points)
        beam_aabb = beam_slice.get_axis_aligned_bounding_box()

        beam_hist, _ = np.histogram(beam_slice_points[:, int(not axis)], bins)
        beam_hist = smooth_histogram(beam_hist, 1)

        low = bins // 2
        high = bins // 2
        for i in range(low, -1, -1):
            if beam_hist[i] > 0:
                low = i
            else:
                break
        for i in range(high, bins):
            if beam_hist[i] > 0:
                high = i
            else:
                break
        slice_width = high - low
        slice_position = slice_width / 2 + low
        print(slice_position / bins)
        print(slice_width / bins)
        beam_slice = get_slice(beam_slice, beam_aabb, int(not axis), slice_position / bins, slice_width / bins, normalized=True)
        beam_aabb = beam_slice.get_axis_aligned_bounding_box()
        beam_aabb.color = (0,1,1) if axis else (1,0,1)
        vis.add_geometry(beam_aabb)


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
    dir_output = dirname + filename + "/"

    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    project_data = shelve.open(dir_output + filename)
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

    vis.run()
    vis.destroy_window()
