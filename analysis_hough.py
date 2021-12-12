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
    :return: Array of layers corresponding to beam directions
    """

    timer.start("Hough Analysis")

    dir_hough = ui.dir_output + "hough/"

    if not os.path.exists(dir_hough):
        os.makedirs(dir_hough)

    scale = 8

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

    render_lines(output_raw, lines)

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

    lines_h = join_lines(lines_h, 0, on_dist=10)
    lines_v = join_lines(lines_v, 1, on_dist=10)

    lines_h = join_lines(lines_h, 0, on_dist=50)
    lines_v = join_lines(lines_v, 1, on_dist=50)

    lines_h = [line for line in lines_h if line_length(line) > 50]
    lines_v = [line for line in lines_v if line_length(line) > 50]

    render_lines(output_joined, lines_h, line_color=(255, 0, 0))
    render_lines(output_joined, lines_v, line_color=(0, 255, 0))
    render_lines(output_joined, lines_other, line_color=(255, 0, 255))

    # Detect beam positions
    clusters_h = cluster_lines(lines_h, 0)
    clusters_v = cluster_lines(lines_v, 1)

    layer_h = BeamSystemLayer()
    for cluster in clusters_h:
        if beam := cluster_to_beam(cluster, scale, aabb, 0):
            layer_h.add_beam(beam)
    layer_h.finalize()

    layer_v = BeamSystemLayer()
    for cluster in clusters_v:
        if beam := cluster_to_beam(cluster, scale, aabb, 1):
            layer_v.add_beam(beam)
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


def cluster_to_beam(cluster, scale, aabb, axis):
    min_bound, max_bound = get_cluster_extents(cluster)
    min_bound *= scale
    max_bound *= scale
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


def render_lines(img, lines, line_color=(255, 0, 0), end_color=(0, 0, 255)):
    # Draw lines
    for line in lines:
        p0, p1 = line
        cv2.line(img, p0, p1, line_color, 2)

    # Draw line endpoints
    for line in lines:
        p0, p1 = line
        cv2.circle(img, p0, 2, end_color, -1)
        cv2.circle(img, p1, 2, end_color, -1)


def cluster_lines(lines, axis):
    not_axis = int(not axis)
    lengths = [line_length(line) for line in lines]
    clusters = []
    dist = 50

    lines = [x for _, x in sorted(zip(lengths, lines))]
    lines.reverse()

    clusters.append([])
    clusters[-1].append(lines.pop())

    while len(lines):
        line = lines.pop()
        for i, cluster in enumerate(clusters):
            l2 = cluster[0]
            if axdiff(line[0],l2[0],not_axis) < dist:
                clusters[i].append(line)
                break
        clusters.append([line])

    return clusters


def get_cluster_extents(cluster):
    x_vals = []
    x_vals += [line[0][0] for line in cluster]
    x_vals += [line[1][0] for line in cluster]

    y_vals = []
    y_vals += [line[0][1] for line in cluster]
    y_vals += [line[1][1] for line in cluster]

    min_bound = np.array((min(x_vals), min(y_vals), 0))
    max_bound = np.array((max(x_vals), max(y_vals), 0))

    return min_bound, max_bound


def line_length(line):
    p0, p1 = line
    d = math.sqrt(pow(p0[0] - p1[0], 2) + pow(p0[1] - p1[1], 2))
    return d


def join_lines(lines, axis, on_dist=10):
    not_axis = int(not axis)
    counter = 0
    off_dist = 10

    while True:
        flag = False
        for i, a in enumerate(lines):
            if flag:
                break
            for j, b in enumerate(lines):
                if a == b:
                    continue
                if flag:
                    break
                if axdiff(a[0], b[0], not_axis) < off_dist:
                    if min((axdiff(a[0], b[0], axis), axdiff(a[1], b[1], axis), axdiff(a[0], b[1], axis), axdiff(a[1], b[0], axis))) < on_dist:
                        new_shift = a[0][not_axis] if axdiff(a[0], a[1], axis) > axdiff(b[0], b[1], axis) else b[0][not_axis]
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

        if not flag:
            break

    print("Joined in {} steps".format(counter))
    return lines


def axdiff(a, b, axis):
    return abs(a[axis] - b[axis])

