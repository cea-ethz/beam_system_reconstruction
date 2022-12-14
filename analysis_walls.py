import math
import open3d as o3d
import numpy as np
import scipy.signal as signal

import main
import settings
import ui
import util_alpha_shape
import util_cloud
import util_histogram
import util_scaling_density


bin_width = 50


def analyze_walls(pc, aabb):
    points = np.asarray(pc.points)
    print("Detecting Walls : ")
    bin_count_x = math.ceil(aabb.get_extent()[0] / bin_width)
    hist_x, bin_edges = np.histogram(points[:, 0], bin_count_x)
    hist_x, hist_x_smooth = util_histogram.process_histogram(hist_x)
    peaks_x, properties = signal.find_peaks(hist_x_smooth, width=1, prominence=0.4)
    print(" X Peaks {} : ".format(peaks_x))

    bin_count_y = math.ceil(aabb.get_extent()[1] / bin_width)
    hist_y, bin_edges = np.histogram(points[:, 1], bin_count_y)
    hist_y, hist_y_smooth = util_histogram.process_histogram(hist_y)
    peaks_y, properties = signal.find_peaks(hist_y_smooth, width=1, prominence=0.4)
    print(" Y Peaks {} : ".format(peaks_y))

    util_histogram.render_bar(ui.axs[0, 1], hist_x, hist_x_smooth, peaks_x)
    util_histogram.render_bar(ui.axs[0, 2], hist_y, hist_y_smooth, peaks_y)

    #axs[1,0].axis(xmin=-10000,xmax=10000,ymin=-1000,ymax=7000)

    # Was -1 for a while, maybe necessary while the histo anlalysis is still weird
    offset = 0

    for peak_x in peaks_x:
        pc = handle_peak(pc, aabb, peak_x + offset, hist_x_smooth, bin_count_x, 0)

    for peak_y in peaks_y:
        pc = handle_peak(pc, aabb, peak_y + offset, hist_y_smooth, bin_count_y, 1)

    return pc


def handle_peak(pc, aabb, peak, hist, bin_count, axis):
    #np.savetxt("hist_{}.txt".format(peak), hist)
    position, width = util_histogram.get_peak_slice_params(hist, peak, diff=0.2)
    position += 0.5
    width += 1.5
    #print("{} : {} : {}".format(peak, position / bin_count, width / bin_count))
    #print(hist)
    peak_slice = util_cloud.get_slice(pc, aabb, axis, position / bin_count, width / bin_count, normalized=True)
    peak_slice.paint_uniform_color((1, 0, 0))
    #vis.add_geometry(peak_slice)
    slice_points = np.asarray(peak_slice.points)
    mean = np.median(slice_points[:, axis])
    # TK NB if the split width is raised to 30 from 20 the dag drawing breaks?
    interior, exterior = util_cloud.split_slice(pc, aabb, axis, mean, 40, normalized=False)
    interior_points = np.asarray(interior.points)

    interior_points = util_cloud.flatten_to_axis(interior_points, axis)

    #if util_alpha_shape.analyze_alpha_shape_density2(interior_points, settings.read("tuning.wall_fill_cutoff"), "wall_{}_{}.png".format(axis,peak)):
    if util_scaling_density.compute_scaling_density(interior_points, "wall_{}_{}".format(axis, peak)) > settings.read("tuning.wall_fill_cutoff"):
        #interior.paint_uniform_color((0.5,0.5,0.5))
        interior.paint_uniform_color(main.color_wall)
        if settings.read("visibility.walls_extracted"):
            #ui.vis.add_geometry(interior)
            aabb = interior.get_axis_aligned_bounding_box()
            main.add_mesh_from_aabb(aabb,main.color_wall)
        return exterior
    else:
        return pc
