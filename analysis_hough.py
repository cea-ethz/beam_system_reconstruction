import cv2
import math
import numpy as np
import open3d as o3d
import os
import skimage.feature
import skimage.transform

import settings
import timer
import ui

from BIM_Geometry import Beam, BeamSystemLayer
from util_cloud import cloud_to_accumulator


def analyze_by_hough_transform(pc, aabb, name="_"):
    """
    Analyzes a beam layer by 2d hough-transform

    :param pc: Pointcloud representing the guessed beam layer
    :param aabb: Bounding box for the point cloud
    :param name: Name for output file
    :return: Array of layers corresponding to beam directions
    """

    timer.start("Hough Analysis")

    dir_hough = ui.dir_output + "hough/"

    if not os.path.exists(dir_hough):
        os.makedirs(dir_hough)

    scale = 16

    accumulator = cloud_to_accumulator(np.array(pc.points), scale)
    cv2.imwrite(dir_hough + name + "_accumulator_raw.png", accumulator)

    ret, accumulator = cv2.threshold(accumulator, 22, 255, cv2.THRESH_BINARY)

    cv2.imwrite(dir_hough + name + "_accumulator_threshold.png", accumulator)

    output_raw = cv2.cvtColor(accumulator, cv2.COLOR_GRAY2BGR)
    output_joined = np.copy(output_raw)

    edges = skimage.feature.canny(accumulator, 2, 1, 25)
    lines = skimage.transform.probabilistic_hough_line(edges, threshold=10, line_length=5, line_gap=3)

    if settings.read("verbosity.global_level") == 2:
        print("{} lines found".format(len(lines)))

    _render_lines(output_raw, lines)

    # Output raw lines graphic
    output_raw = output_raw.astype(np.uint8)
    cv2.imwrite(dir_hough + name + "_lines_raw.png", output_raw)

    # Find final outlines
    lines_h = []
    lines_v = []
    lines_other = []

    ratio = 5

    # Divide between horizontal and vertical
    for line in lines:
        p0, p1 = line
        dx = abs(p0[0] - p1[0])
        dy = abs(p0[1] - p1[1])
        if dy == 0 or dx / dy > ratio:
            lines_h.append(line)
        elif dx == 0 or dy / dx > ratio:
            lines_v.append(line)
        else:
            lines_other.append(line)

    # Join lines and filter out remaining too-short lines

    lines_h = _join_lines(lines_h, 0, on_dist=10)
    lines_v = _join_lines(lines_v, 1, on_dist=10)

    lines_h = _join_lines(lines_h, 0, on_dist=50)
    lines_v = _join_lines(lines_v, 1, on_dist=50)

    lines_h = [line for line in lines_h if _line_length(line) > 50]
    lines_v = [line for line in lines_v if _line_length(line) > 50]

    _render_lines(output_joined, lines_h, line_color=(255, 0, 0))
    _render_lines(output_joined, lines_v, line_color=(0, 255, 0))
    _render_lines(output_joined, lines_other, line_color=(255, 0, 255))

    # Detect beam positions
    clusters_h = _cluster_lines(lines_h, 0, scale)
    clusters_v = _cluster_lines(lines_v, 1, scale)

    layer_h = BeamSystemLayer()
    for cluster in clusters_h:
        if beam := _cluster_to_beam(cluster, scale, aabb, 0):
            layer_h.add_beam(beam)

    layer_v = BeamSystemLayer()
    for cluster in clusters_v:
        if beam := _cluster_to_beam(cluster, scale, aabb, 1):
            layer_v.add_beam(beam)

    # Discard beams with too few points for their volume, assumed to be false positives
    eps = np.finfo(float).eps
    density_h = []
    for beam in layer_h.beams:
        beam.cloud = pc.crop(beam.aabb)
        density_h.append(beam.get_density())
    density_h = np.array(density_h)
    median_h = np.median(density_h)
    layer_h.beams = [beam for beam in layer_h.beams if median_h / (beam.get_density() + eps) < 10]

    density_v = []
    for beam in layer_v.beams:
        beam.cloud = pc.crop(beam.aabb)
        density_v.append(beam.get_density())
    density_v = np.array(density_v)
    median_v = np.median(density_v)
    layer_v.beams = [beam for beam in layer_v.beams if median_v / (beam.get_density() + eps) < 10]

    print(f"Initial H Count : {len(layer_h.beams)}")
    print(f"Initial V Count : {len(layer_v.beams)}")

    # Extend Beams as Necessary for Later Splitting
    layer_h = _extend_layer(layer_h, layer_v, 1)
    layer_v = _extend_layer(layer_v, layer_h, 0)

    # Finalize Beam Layers
    layer_h.finalize()
    layer_v.finalize()

    output_joined = output_joined.astype(np.uint8)
    cv2.imwrite(dir_hough + name + "_lines_joined.png", output_joined)

    if settings.read("display.hough"):
        timer.pause("Hough Analysis")
        cv2.imshow("hough", output_joined)
        cv2.waitKey()
        cv2.destroyAllWindows()
        timer.unpause("Hough Analysis")

    timer.end("Hough Analysis")

    return [layer_h, layer_v]


def _extend_layer(layer_a, layer_b, axis):
    not_axis = int(not axis)

    d = 200

    for beam_id, beam in enumerate(layer_a.beams):
        center = beam.aabb.get_center()
        half_extent = beam.aabb.get_half_extent()
        pa = np.copy(center)
        pb = np.copy(center)

        pa[axis] -= half_extent[axis]
        pb[axis] += half_extent[axis]

        min_bound = beam.aabb.get_min_bound()
        max_bound = beam.aabb.get_max_bound()

        for beam2 in layer_b.beams:
            center2 = beam2.aabb.get_center()
            half_extent2 = beam2.aabb.get_half_extent()
            pa2 = np.copy(center2)
            pb2 = np.copy(center2)

            pa2[not_axis] -= half_extent2[not_axis]
            pb2[not_axis] += half_extent2[not_axis]

            side = center2[axis] < center[axis]
            start = pa if side else pb
            if abs(center2[axis] - start[axis]) > d:
                continue
            if (pa2[not_axis] - d) <= start[not_axis] <= (pb2[not_axis] + d):
                if side:
                    min_bound[axis] = center2[axis]
                else:
                    max_bound[axis] = center2[axis]

        beam.aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    return layer_a


def _render_lines(img, lines, line_color=(255, 0, 0), end_color=(0, 0, 255)):
    # Draw lines
    for line in lines:
        p0, p1 = line
        cv2.line(img, p0, p1, line_color, 2)

    # Draw line endpoints
    for line in lines:
        p0, p1 = line
        cv2.circle(img, p0, 2, end_color, -1)
        cv2.circle(img, p1, 2, end_color, -1)


def _cluster_lines(lines, axis, scale):
    """
    Join together lines that likely belong to opposite sides of the same beam

    :param lines:
    :param axis:
    :return:
    """
    not_axis = int(not axis)
    lengths = [_line_length(line) for line in lines]
    clusters = []
    dist = 400 / scale

    lines = [x for _, x in sorted(zip(lengths, lines))]
    lines.reverse()

    clusters.append([])
    clusters[-1].append(lines.pop())

    while len(lines):
        line = lines.pop()
        for i, cluster in enumerate(clusters):
            l2 = cluster[0]
            if _axdiff(line[0], l2[0], not_axis) < dist:
                clusters[i].append(line)
                #print(f"Axis : {axis},cluster {i},  line {line} clustered to line {l2}")
                break
        clusters.append([line])

    return clusters


def _cluster_to_beam(cluster, scale, aabb, axis):
    min_bound, max_bound = _get_cluster_extents(cluster)
    min_bound *= scale
    max_bound *= scale
    # Shift elements, as image origin was at corner of bounding box
    min_bound[0] += aabb.get_min_bound()[1]
    min_bound[1] += aabb.get_min_bound()[0]
    max_bound[0] += aabb.get_min_bound()[1]
    max_bound[1] += aabb.get_min_bound()[0]
    min_bound[0], min_bound[1] = min_bound[1], min_bound[0]
    max_bound[0], max_bound[1] = max_bound[1], max_bound[0]
    if min_bound[0] == max_bound[0] or min_bound[1] == max_bound[1]:
        return None
    min_bound[2] = aabb.get_min_bound()[2]
    max_bound[2] = aabb.get_max_bound()[2]
    beam_aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    beam = Beam(beam_aabb, axis, None)
    return beam


def _get_cluster_extents(cluster):
    x_vals = []
    x_vals += [line[0][0] for line in cluster]
    x_vals += [line[1][0] for line in cluster]

    y_vals = []
    y_vals += [line[0][1] for line in cluster]
    y_vals += [line[1][1] for line in cluster]

    min_bound = np.array((min(x_vals), min(y_vals), 0))
    max_bound = np.array((max(x_vals), max(y_vals), 0))

    return min_bound, max_bound


def _line_length(line):
    p0, p1 = line
    d = math.sqrt(pow(p0[0] - p1[0], 2) + pow(p0[1] - p1[1], 2))
    return d


def _join_lines(lines, axis, on_dist=10):
    not_axis = int(not axis)
    counter = 0
    off_dist = 10

    start = 0

    while True:
        flag = False
        for i in range(start, len(lines)):
            if flag:
                break
            a = lines[i]
            for j in range(start, len(lines)):
                if i == j:
                    continue
                if flag:
                    break

                b = lines[j]
                # Check if segments are roughly on same true line
                if _axdiff(a[0], b[0], not_axis) < off_dist:
                    if min((_axdiff(a[0], b[0], axis), _axdiff(a[1], b[1], axis), _axdiff(a[0], b[1], axis), _axdiff(a[1], b[0], axis))) < on_dist:
                        new_shift = a[0][not_axis] if _axdiff(a[0], a[1], axis) > _axdiff(b[0], b[1], axis) else b[0][not_axis]
                        min_new = min((a[0][axis], a[1][axis], b[0][axis], b[1][axis]))
                        max_new = max((a[0][axis], a[1][axis], b[0][axis], b[1][axis]))
                        p0 = (min_new, new_shift) if not axis else (new_shift, min_new)
                        p1 = (max_new, new_shift) if not axis else (new_shift, max_new)
                        new_line = (p0, p1)
                        if i < j:
                            del lines[j]
                            del lines[i]
                        else:
                            del lines[i]
                            del lines[j]

                        lines.append(new_line)

                        flag = True
                        counter += 1

            # End inner loop
            if not flag and i == start:
                start += 1

        # End outer loop

        if not flag:
            break

    print("Joined in {} steps".format(counter))
    return lines


def _axdiff(pa, pb, axis):
    """
    Returns the distance between two points only along the specified axis

    :param pa:
    :param pb:
    :param axis:
    :return:
    """
    return abs(pa[axis] - pb[axis])



