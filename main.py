import math
import networkx as nx
import numpy as np
import open3d as o3d
import os
import progressbar
import scipy.signal as signal
import shelve
import time

from matplotlib import pyplot as plt
from tkinter import filedialog

import util_histogram

from BIM_Geometry import Beam, BeamSystemLayer


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

# Intended bin size in mm
bin_width = 50

beam_layers = []

DG = nx.DiGraph()


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





def get_peak_slice_params(hist, peak, diff=0.1):
    """
    Given a peak, return the slice parameters to stay within [diff] of that peak (may not be centered at the peak)
    :param hist:
    :param peak:
    :param diff:
    :return:
    """
    low = peak
    high = peak+1

    for i in range(low - 1, -1, -1):
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

    z_bins = math.ceil(aabb.get_extent()[2] / bin_width)

    hist_z, bin_edges = np.histogram(points[:, 2], z_bins)
    hist_z, hist_z_smooth = util_histogram.process_histogram(hist_z)
    mean_z = np.mean(hist_z_smooth)

    peaks, properties = signal.find_peaks(hist_z_smooth, width=1, prominence=0.1)

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
        pc_slice = get_slice(pc, aabb, 2, peak_slice_position / z_bins, peak_slice_width / z_bins, normalized=True)
        pc_slice_aabb = pc_slice.get_axis_aligned_bounding_box()
        slice_points = np.asarray(pc_slice.points)

        rel_height = 0.75  # Check the width near the bottom of the peak
        prominence = 0.13  # Experimentally tuned, this should be determined more exactly
        peak_width = 4
        # Take histogram along X axis
        bin_count_x = math.ceil(pc_slice_aabb.get_extent()[0] / bin_width)
        #print("Bin Count X : {}".format(bin_count_x))
        hist_x, _ = np.histogram(slice_points[:, 0], bin_count_x)
        hist_x, hist_x_smooth = util_histogram.process_histogram(hist_x)
        mean_x = np.mean(hist_x_smooth)
        peaks_x, _ = signal.find_peaks(hist_x_smooth, width=peak_width, prominence=prominence, rel_height=rel_height)

        # Take histogram along Y Axis
        bin_count_y = math.ceil(pc_slice_aabb.get_extent()[1] / bin_width)
        #print("Bin Count Y : {}".format(bin_count_y))
        hist_y, _ = np.histogram(slice_points[:, 1], bin_count_y)
        hist_y, hist_y_smooth = util_histogram.process_histogram(hist_y)
        mean_y = np.mean(hist_y_smooth)
        peaks_y, _ = signal.find_peaks(hist_y_smooth, width=peak_width, prominence=prominence, rel_height=rel_height)

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

        print("Peak : {}, Variance X : {}, Variance Y : {}".format(peak, variance_x, variance_y))

        if variance_x < variance_split or variance_y < variance_split:
            o3d.io.write_point_cloud(dir_output + filename + "_grid_{}.ply".format(peak), pc_slice)
            analyze_beam_system(pc_slice, pc_slice_aabb, 0, hist_x_smooth, peaks_x, bin_count_x)
            analyze_beam_system(pc_slice, pc_slice_aabb, 1, hist_y_smooth, peaks_y, bin_count_y)


def analyze_beam_system(pc, aabb, axis, hist, peaks, source_bin_count):
    not_axis = int(not axis)
    beam_layers.append(BeamSystemLayer())
    for peak in peaks:
        slice_position, slice_width = get_peak_slice_params(hist, peak, 0.1)
        #print("beam!")
        #print(slice_position)
        #print(slice_width)
        beam_slice = get_slice(pc, aabb, axis, slice_position / source_bin_count, slice_width / source_bin_count, normalized=True)
        beam_slice_points = np.array(beam_slice.points)
        beam_aabb = beam_slice.get_axis_aligned_bounding_box()
        extent = beam_aabb.get_extent()

        bin_count = math.ceil(beam_aabb.get_extent()[not_axis] / bin_width)
        #print("Bin Count : {}".format(bin_count))

        beam_hist, _ = np.histogram(beam_slice_points[:, not_axis], bin_count)
        beam_hist = util_histogram.smooth_histogram(beam_hist, 1)

        # Count out from the median value
        median = np.median(beam_slice_points[:, not_axis])
        median_bin = int((median - (beam_aabb.get_center()[not_axis] - beam_aabb.get_half_extent()[not_axis])) / beam_aabb.get_extent()[not_axis] * bin_count)
        #print("Median {} in {} to {}".format(median,beam_aabb.get_center()[not_axis] - beam_aabb.get_half_extent()[not_axis],beam_aabb.get_center()[not_axis] + beam_aabb.get_half_extent()[not_axis]))
        #print("Median bin : " + str(median_bin))
        low = median_bin
        high = median_bin + 1
        for i in range(low, -1, -1):
            if beam_hist[i] > 0:
                low = i
            else:
                break
        for i in range(high, bin_count):
            if beam_hist[i] > 0:
                high = i
            else:
                break
        slice_width = high - low
        slice_position = slice_width / 2 + low

        beam_slice = get_slice(beam_slice, beam_aabb, int(not axis), slice_position / bin_count, slice_width / bin_count, normalized=True)
        beam_aabb = beam_slice.get_axis_aligned_bounding_box()
        beam_aabb.color = (0, 1, 1) if axis else (1, 0, 1)

        beam_layers[-1].add_beam(Beam(beam_aabb, axis, beam_slice))
        #vis.add_geometry(beam_slice)
        #vis.add_geometry(beam_aabb)
    beam_layers[-1].finalize()


def setup_vis():
    vis = o3d.visualization.VisualizerWithKeyCallback()
    #vis.register_key_callback(83, save_view)
    vis.register_key_callback(69, set_up_vector)
    vis.create_window()
    vis.get_render_option().point_size = 1

    return vis


def perform_beam_splits(primary_layer, secondary_layer):
    # Perhaps we need to check for actual intersection first but for now theres no issues
    new_secondary = BeamSystemLayer()
    while len(secondary_layer.beams) > 0:
        sb = secondary_layer.beams.pop()
        flag = False
        for pb in primary_layer.beams:
            location = sb.get_point_param(pb.aabb.get_center())
            if 0 < location < sb.length:
                if 0.1 < location < (sb.length - 0.1):
                    flag = True
                    new_a, new_b = sb.split(location)
                    if new_a.length > 500:
                        secondary_layer.beams.append(new_a)
                    if new_b.length > 500:
                        secondary_layer.beams.append(new_b)

                    break
        if not flag:
            new_secondary.add_beam(sb)
    print(new_secondary.beams)
    return new_secondary


def analyze_beam_connections(primary_layer,secondary_layer):
    # Create edges
    for sb in secondary_layer.beams:
        for pb in primary_layer.beams:
            if sb.check_overlap(pb):
                DG.add_edges_from([(sb.id, pb.id)])

    # Set node_layers
    for pb in primary_layer.beams:
        if pb.id in DG.nodes:
            DG.nodes[pb.id]['layer'] = 0
    for sb in secondary_layer.beams:
        if sb.id in DG.nodes:
            DG.nodes[sb.id]['layer'] = 1


def get_stream_counts(dg, id):
    upstream = [n for n in nx.traversal.bfs_tree(dg, id, reverse=True) if n != id]
    downstream = [n for n in nx.traversal.bfs_tree(dg, id) if n != id]
    return len(upstream), len(downstream)

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
    print(cloud)
    vis.add_geometry(cloud)

    aabb_main = cloud.get_axis_aligned_bounding_box()
    aabb_main.color = (1, 0, 0)
    vis.add_geometry(aabb_main)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000, origin=[0, 0, 0])
    vis.add_geometry(mesh_frame)

    analyze_z_levels(cloud, aabb_main)
    plt.savefig(dir_output + filename + "_plot.png")
    plt.show()

    if len(beam_layers) > 2:
        print("Error : Handling more than 2 beam layers not yet implemented")
    primary_id = int(beam_layers[0].mean_spacing < beam_layers[1].mean_spacing)
    secondary = perform_beam_splits(beam_layers[primary_id], beam_layers[int(not primary_id)])

    for beam in beam_layers[primary_id].beams:
        vis.add_geometry(beam.cloud)
        vis.add_geometry(beam.aabb)
    for beam in secondary.beams:
        vis.add_geometry(beam.cloud)
        vis.add_geometry(beam.aabb)

    analyze_beam_connections(beam_layers[primary_id],secondary)

    pos = nx.multipartite_layout(DG, 'layer')
    for i,pb in enumerate(beam_layers[primary_id].beams):
        n = 1.0 * i / (len(beam_layers[primary_id].beams) - 1)
        pos[pb.id][1] = n * 2 - 1

    # Calculate downstream
    for beam in beam_layers[primary_id].beams:
        if beam.id not in DG.nodes:
            continue
        upstream, downstream = get_stream_counts(DG,beam.id)
        DG.nodes[beam.id]['stream'] = upstream

    for beam in secondary.beams:
        if beam.id not in DG.nodes:
            continue
        upstream, downstream = get_stream_counts(DG,beam.id)
        DG.nodes[beam.id]['stream'] = upstream

    labels = nx.get_node_attributes(DG,'stream')
    nx.draw(DG, pos, labels=labels, with_labels=True,node_size=300)
    plt.savefig(dir_output + filename + "_graph.png")
    plt.show()

    vis.remove_geometry(cloud)

    vis.run()
    vis.destroy_window()
