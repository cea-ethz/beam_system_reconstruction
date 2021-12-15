import math
import numpy as np
import open3d as o3d
import scipy.signal as signal

import analysis_hough
import settings
import timer
import ui
import util_alpha_shape
import util_cloud
import util_histogram

from BIM_Geometry import Beam, BeamSystemLayer

# Intended bin size in mm
bin_width = 50

variance_split = 0.075

dumb_flag = False


def detect_beams(pc, aabb):
    points = np.asarray(pc.points)

    bin_count_z = math.ceil(aabb.get_extent()[2] / bin_width)

    hist_z, bin_edges = np.histogram(points[:, 2], bin_count_z)
    hist_z, hist_z_smooth = util_histogram.process_histogram(hist_z, extension=2)

    peaks, properties = signal.find_peaks(hist_z_smooth, width=1, prominence=0.1)

    if len(peaks) == 0:
        print("Error : No Z peaks found")
        print(hist_z)

    util_histogram.render_bar(ui.axs[0, 0], hist_z, hist_z_smooth, peaks)

    beam_layers = []
    column_slice_positions = []

    floor_levels = []

    for i, peak in enumerate(peaks):
        # Get extents of peak
        # TODO: the falloff here needs to be calculated from cloud (e.g. needs to be 0.1 for pg, and 0.25 for gt)
        peak_slice_position, peak_slice_width = util_histogram.get_peak_slice_params(hist_z_smooth, peak, settings.read("tuning.beam_z_falloff"))

        # Get slice at Z height
        pc_slice = util_cloud.get_slice(pc, aabb, 2, peak_slice_position / bin_count_z, peak_slice_width / bin_count_z, normalized=True)
        pc_slice_aabb = pc_slice.get_axis_aligned_bounding_box()

        new_levels = _analyze_z_level(pc_slice, pc_slice_aabb, peak)
        beam_layers += new_levels

        # If the peak is a beam system, record the real position 1 meter below the slice to start analyzing for columns
        if len(new_levels):
            column_slice_positions.append(bin_edges[peak] - 1000)
        else:
            floor_levels.append(peak_slice_position * bin_width + aabb.get_min_bound()[2])

    return beam_layers, column_slice_positions, floor_levels


def _analyze_z_level(pc, aabb, peak):
    slice_points = np.asarray(pc.points)

    rel_height = 0.75  # Check the width near the bottom of the peak
    prominence = 0.13  # Experimentally tuned, this should be determined more exactly
    peak_width = 4
    padding = 3
    # Take histogram along X axis
    bin_count_x = math.ceil(aabb.get_extent()[0] / bin_width)
    # print("Bin Count X : {}".format(bin_count_x))
    hist_x, _ = np.histogram(slice_points[:, 0], bin_count_x)
    hist_x = np.pad(hist_x, (padding, padding), 'constant', constant_values=(0, 0))
    hist_x, hist_x_smooth = util_histogram.process_histogram(hist_x)
    mean_x = np.mean(hist_x_smooth)
    peaks_x, _ = signal.find_peaks(hist_x_smooth, width=peak_width, prominence=prominence, rel_height=rel_height)
    # Undo padding
    peaks_x -= padding
    hist_x = hist_x[padding:-padding]
    hist_x_smooth = hist_x_smooth[padding:-padding]

    # Take histogram along Y Axis
    bin_count_y = math.ceil(aabb.get_extent()[1] / bin_width)
    # print("Bin Count Y : {}".format(bin_count_y))
    hist_y, _ = np.histogram(slice_points[:, 1], bin_count_y)
    hist_y = np.pad(hist_y, (padding, padding), 'constant', constant_values=(0, 0))
    hist_y, hist_y_smooth = util_histogram.process_histogram(hist_y)
    mean_y = np.mean(hist_y_smooth)
    peaks_y, _ = signal.find_peaks(hist_y_smooth, width=peak_width, prominence=prominence, rel_height=rel_height)
    # Undo padding
    peaks_y -= padding
    hist_y = hist_y[padding:-padding]
    hist_y_smooth = hist_y_smooth[padding:-padding]

    alpha_points = util_cloud.flatten_to_axis(slice_points, 2)

    if settings.read("visibility.beam_levels"):
        aabb.color = (0, 1, 0)
        ui.vis.add_geometry(aabb)

    beam_layers = []
    if not util_alpha_shape.analyze_alpha_shape_density2(alpha_points, 0.5, "floor_{}.png".format(peak)):
        # Plot X and Y histograms
        if layer := _analyze_beam_system_layer(pc, aabb, 0, hist_x_smooth, peaks_x, bin_count_x):
            util_histogram.render_bar(ui.axs[1, 1], hist_x, hist_x_smooth, peaks_x)
            beam_layers.append(layer)
        if layer := _analyze_beam_system_layer(pc, aabb, 1, hist_y_smooth, peaks_y, bin_count_y):
            util_histogram.render_bar(ui.axs[1, 2], hist_y, hist_y_smooth, peaks_y)
            beam_layers.append(layer)

    # Logic for which method is used isn't great here,
    if settings.read("analysis.use_hough") and len(beam_layers):
        timer.pause("Beam Analysis")
        beam_layers_hough = analysis_hough.analyze_by_hough_transform(pc, aabb, name=str(peak))
        timer.unpause("Beam Analysis")
        return beam_layers_hough
    else:
        return beam_layers


def _analyze_beam_system_layer(pc, aabb, axis, hist, peaks, source_bin_count):
    not_axis = int(not axis)

    global dumb_flag

    layer = BeamSystemLayer()

    for peak in peaks:
        # 0.15 compromise
        slice_position, slice_width = util_histogram.get_peak_slice_params(hist, peak, 0.1) # This drop either cuts off too much of an end value or allows the other beams to get oo large
        # Drop false positives that are obviously overwide
        if slice_width * bin_width > 1000:
            continue
        beam_slice = util_cloud.get_slice(pc, aabb, axis, slice_position / source_bin_count, slice_width / source_bin_count, normalized=True)
        beam_slice_points = np.array(beam_slice.points)
        beam_aabb = beam_slice.get_axis_aligned_bounding_box()
        aabb_c = beam_aabb.get_center()
        aabb_e = beam_aabb.get_extent()
        aabb_he = beam_aabb.get_half_extent()

        bin_count = math.ceil(aabb_e[not_axis] / bin_width)
        #print("Bin Count : {}".format(bin_count))

        beam_hist, _ = np.histogram(beam_slice_points[:, not_axis], bin_count)
        beam_hist = util_histogram.smooth_histogram(beam_hist, 2)

        # Count out from the median value
        median = np.median(beam_slice_points[:, not_axis])
        median_bin = int((median - (aabb_c[not_axis] - aabb_he[not_axis])) / aabb_e[not_axis] * bin_count)
        #print("Median {} in {} to {}".format(median,beam_aabb.get_center()[not_axis] - beam_aabb.get_half_extent()[not_axis],beam_aabb.get_center()[not_axis] + beam_aabb.get_half_extent()[not_axis]))
        #print("Median bin : " + str(median_bin))
        low = median_bin - 1
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

        if not dumb_flag:
            dumb_flag = True
            util_histogram.render_bar(ui.axs[1, 0], None, beam_hist, [])

        beam_slice = util_cloud.get_slice(beam_slice, beam_aabb, not_axis, slice_position / bin_count, slice_width / bin_count, normalized=True)
        beam_aabb = beam_slice.get_axis_aligned_bounding_box()

        layer.add_beam(Beam(beam_aabb, axis, beam_slice))

    if len(layer.beams):
        layer.finalize()
    else:
        layer = None
    return layer


def perform_beam_splits(primary_layer, secondary_layer):
    # Perhaps we need to check for actual intersection first but for now theres no issues
    new_secondary = BeamSystemLayer()
    while len(secondary_layer.beams) > 0:
        sb = secondary_layer.beams.pop()
        flag = False
        for pb in primary_layer.beams:
            location = sb.get_point_param(pb.aabb.get_center())

            if 0 < location < sb.length:
                if 0.1 < location < (sb.length - 0.1) and sb.check_overlap(pb):

                    split_point = pb.aabb.get_center()
                    split_point[sb.axis] = sb.aabb.get_center()[sb.axis]
                    split_point[2] = split_point[2] + 100

                    if settings.read("visibility.split_points"):
                        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=50)
                        sphere.translate(split_point)
                        sphere.paint_uniform_color((1, 0, 0))
                        ui.vis.add_geometry(sphere)

                    flag = True
                    new_a, new_b = sb.split(location)
                    if new_a.length > 500:
                        secondary_layer.beams.append(new_a)
                    elif settings.read("visibility.beams_rejected"):
                        new_a.aabb.color = (1, 0, 1)
                        ui.vis.add_geometry(new_a.aabb)

                    if new_b.length > 500:
                        secondary_layer.beams.append(new_b)
                    elif settings.read("visibility.beams_rejected"):
                        new_b.aabb.color = (1, 0, 1)
                        ui.vis.add_geometry(new_b.aabb)

                    break
        if not flag:
            new_secondary.add_beam(sb)
    return new_secondary


def analyze_beam_connections(primary_layer, secondary_layer, DG):
    # Create edges
    for sb in secondary_layer.beams:
        for pb in primary_layer.beams:
            if sb.check_overlap(pb):
                DG.add_edges_from([(sb.id, pb.id)])

    # Set node_layers
    for pb in primary_layer.beams:
        if pb.id in DG.nodes:
            DG.nodes[pb.id]['layer'] = 1
            DG.nodes[pb.id]['source'] = 'beam'
    for sb in secondary_layer.beams:
        if sb.id in DG.nodes:
            DG.nodes[sb.id]['layer'] = 2
            DG.nodes[sb.id]['source'] = 'beam'
