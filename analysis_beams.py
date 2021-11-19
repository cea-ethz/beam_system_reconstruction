import math
import numpy as np
import open3d as o3d
import scipy.signal as signal

import util_cloud
import util_histogram

from BIM_Geometry import Beam, BeamSystemLayer
from util_histogram import color_back, color_front, color_front_highlight, color_back_highlight


# Intended bin size in mm
bin_width = 50

variance_split = 0.075

#TK refactor this out of global

def detect_beams(pc, aabb, axs=None):
    points = np.asarray(pc.points)

    bin_count_z = math.ceil(aabb.get_extent()[2] / bin_width)

    hist_z, bin_edges = np.histogram(points[:, 2], bin_count_z)
    hist_z, hist_z_smooth = util_histogram.process_histogram(hist_z)

    peaks, properties = signal.find_peaks(hist_z_smooth, width=1, prominence=0.1)

    util_histogram.render_bar(axs[0, 0], hist_z, hist_z_smooth, peaks)

    beam_layers = []
    column_slice_positions = []

    for i, peak in enumerate(peaks):
        # Get extents of peak
        peak_slice_position, peak_slice_width = util_histogram.get_peak_slice_params(hist_z_smooth, peak, 0.1)

        # Get slice at Z height
        pc_slice = util_cloud.get_slice(pc, aabb, 2, peak_slice_position / bin_count_z, peak_slice_width / bin_count_z, normalized=True)
        pc_slice_aabb = pc_slice.get_axis_aligned_bounding_box()
        beam_layers += _analyze_z_level(pc_slice, pc_slice_aabb, axs)

        # If the peak is a beam system, record the real position 1 meter below the slice to start analyzing for columns
        if len(beam_layers):
            column_slice_positions.append(bin_edges[peak] - 1000)

    return beam_layers, column_slice_positions


def _analyze_z_level(pc, aabb, axs=None):
    slice_points = np.asarray(pc.points)

    rel_height = 0.75  # Check the width near the bottom of the peak
    prominence = 0.13  # Experimentally tuned, this should be determined more exactly
    peak_width = 4
    # Take histogram along X axis
    bin_count_x = math.ceil(aabb.get_extent()[0] / bin_width)
    # print("Bin Count X : {}".format(bin_count_x))
    hist_x, _ = np.histogram(slice_points[:, 0], bin_count_x)
    hist_x, hist_x_smooth = util_histogram.process_histogram(hist_x)
    mean_x = np.mean(hist_x_smooth)
    peaks_x, _ = signal.find_peaks(hist_x_smooth, width=peak_width, prominence=prominence, rel_height=rel_height)

    # Take histogram along Y Axis
    bin_count_y = math.ceil(aabb.get_extent()[1] / bin_width)
    # print("Bin Count Y : {}".format(bin_count_y))
    hist_y, _ = np.histogram(slice_points[:, 1], bin_count_y)
    hist_y, hist_y_smooth = util_histogram.process_histogram(hist_y)
    mean_y = np.mean(hist_y_smooth)
    peaks_y, _ = signal.find_peaks(hist_y_smooth, width=peak_width, prominence=prominence, rel_height=rel_height)

    # Calculate variance on each axis
    variance_x = np.var(hist_x)
    variance_y = np.var(hist_y)

    # print("Peak : {}, Variance X : {}, Variance Y : {}".format(peak, variance_x, variance_y))
    #
    beam_layers = []
    if variance_x < variance_split or variance_y < variance_split:

        # Plot X and Y histograms
        util_histogram.render_bar(axs[1, 1], hist_x, hist_x_smooth, peaks_x)
        util_histogram.render_bar(axs[1, 2], hist_y, hist_y_smooth, peaks_y)

        #o3d.io.write_point_cloud(dir_output + filename + "_grid_{}.ply".format(0), pc)
        beam_layers.append(_analyze_beam_system_layer(pc, aabb, 0, hist_x_smooth, peaks_x, bin_count_x))
        beam_layers.append(_analyze_beam_system_layer(pc, aabb, 1, hist_y_smooth, peaks_y, bin_count_y))
    return beam_layers


def _analyze_beam_system_layer(pc, aabb, axis, hist, peaks, source_bin_count):
    not_axis = int(not axis)

    layer = BeamSystemLayer()

    for peak in peaks:
        slice_position, slice_width = util_histogram.get_peak_slice_params(hist, peak, 0.1)
        #print("beam!")
        #print(slice_position)
        #print(slice_width)
        beam_slice = util_cloud.get_slice(pc, aabb, axis, slice_position / source_bin_count, slice_width / source_bin_count, normalized=True)
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

        beam_slice = util_cloud.get_slice(beam_slice, beam_aabb, int(not axis), slice_position / bin_count, slice_width / bin_count, normalized=True)
        beam_aabb = beam_slice.get_axis_aligned_bounding_box()
        #beam_aabb.color = (0, 1, 1) if axis else (1, 0, 1)

        layer.add_beam(Beam(beam_aabb, axis, beam_slice))
        #vis.add_geometry(beam_slice)
        #vis.add_geometry(beam_aabb)
    layer.finalize()
    return layer


def perform_beam_splits(primary_layer, secondary_layer, vis=None):
    # Perhaps we need to check for actual intersection first but for now theres no issues
    new_secondary = BeamSystemLayer()
    while len(secondary_layer.beams) > 0:
        sb = secondary_layer.beams.pop()
        flag = False
        for pb in primary_layer.beams:
            location = sb.get_point_param(pb.aabb.get_center())

            if 0 < location < sb.length:
                if 0.1 < location < (sb.length - 0.1):

                    split_point = pb.aabb.get_center()
                    split_point[sb.axis] = sb.aabb.get_center()[sb.axis]
                    split_point[2] = split_point[2] + 100
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=50)
                    sphere.translate(split_point)
                    sphere.paint_uniform_color((1,0,0))
                    vis.add_geometry(sphere)

                    flag = True
                    new_a, new_b = sb.split(location)
                    if new_a.length > 500:
                        secondary_layer.beams.append(new_a)
                    else:
                        new_a.aabb.color = (1,0,1)
                        vis.add_geometry(new_a.aabb)
                    if new_b.length > 500:
                        secondary_layer.beams.append(new_b)
                    else:
                        new_b.aabb.color = (1, 0, 1)
                        vis.add_geometry(new_b.aabb)

                    break
        if not flag:
            new_secondary.add_beam(sb)
    return new_secondary


def analyze_beam_connections(primary_layer, secondary_layer,DG):
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