import math


def check_element_counts(data_gt, data_scan):
    columns_gt, beams_p_gt, beams_s_gt = _count_types(data_gt)
    columns_scan, beams_p_scan, beams_s_scan = _count_types(data_scan)

    return columns_scan - columns_gt, (beams_p_scan - beams_p_gt) + (beams_s_scan - beams_s_gt)


def check_column_quality(data_gt, data_scan):
    column_diffs_cs_offset = []
    column_diffs_cs_size = []
    column_diffs_length = []
    column_centers_gt = _get_column_centers(data_gt)
    column_centers_scan = _get_column_centers(data_scan)

    column_dims_gt = _get_column_dims(data_gt)
    column_dims_scan = _get_column_dims(data_scan)

    while len(column_centers_scan):
        center_scan = column_centers_scan.pop()
        center_dims = column_dims_scan.pop()
        best_id = -1
        best_dist = -1
        for i, center_gt in enumerate(column_centers_gt):
            dist = math.dist(center_scan, center_gt)
            if best_id == -1 or dist < best_dist:
                best_id = i
                best_dist = dist
        column_diffs_cs_offset.append(best_dist)
        column_diffs_length.append(abs(center_dims[2] - column_dims_gt[best_id][2]))
        column_diffs_cs_size.append(abs(center_dims[0] - column_dims_gt[best_id][0]) + abs(center_dims[1] - column_dims_gt[best_id][1]))
        del column_centers_gt[best_id]
        del column_dims_gt[best_id]

    column_cs_offset_average = sum(column_diffs_cs_offset) / len(column_diffs_cs_offset)
    column_cs_size_average = sum(column_diffs_cs_size) / len(column_diffs_cs_size)
    column_length_average = sum(column_diffs_length) / len(column_diffs_length)

    return column_cs_offset_average, column_cs_size_average, column_length_average


def check_beam_quality(data_gt, data_scan):
    beam_diffs_cs_offsets = []
    beam_diffs_cs_size = []
    beam_diffs_length = []

    new_offsets, new_cs, new_lengths = _get_beam_layer_diffs(data_gt, data_scan, 0)
    beam_diffs_cs_offsets += new_offsets
    beam_diffs_cs_size += new_cs
    beam_diffs_length += new_lengths
    new_offsets, new_cs, new_lengths = _get_beam_layer_diffs(data_gt, data_scan, 1)
    beam_diffs_cs_offsets += new_offsets
    beam_diffs_cs_size += new_cs
    beam_diffs_length += new_lengths

    print(beam_diffs_length)
    print(beam_diffs_cs_size)

    beam_cs_offset_average = sum(beam_diffs_cs_offsets) / len(beam_diffs_cs_offsets)
    beam_cs_size_average = sum(beam_diffs_cs_size) / len(beam_diffs_cs_size)
    beam_length_diff_average = sum(beam_diffs_length) / len(beam_diffs_length)
    return beam_cs_offset_average, beam_cs_size_average, beam_length_diff_average


def _get_column_centers(csv):
    centers = []
    for line in csv:
        parts = line.split(",")
        if parts[0] != "column":
            continue
        center = [float(parts[2]), float(parts[3])]
        centers.append(center)
    return centers


def _get_column_dims(csv):
    dims = []
    for line in csv:
        parts = line.split(",")
        if parts[0] != "column":
            continue
        dim = [float(parts[5]), float(parts[6]), float(parts[7])]
        dims.append(dim)
    return dims


def _get_beam_layer_diffs(data_gt, data_scan, axis):
    beam_cs_offset_diffs = []
    beam_cs_size_diffs = []
    beam_length_diffs = []
    beam_centers_gt_2d = _get_beam_centers_2d(data_gt, axis)
    beam_centers_scan_2d = _get_beam_centers_2d(data_scan, axis)

    beam_centers_gt_3d = _get_beam_centers_2d(data_gt, axis)
    beam_centers_scan_3d = _get_beam_centers_2d(data_scan, axis)

    beam_dims_gt = _get_beam_dims(data_gt, axis)
    beam_dims_scan = _get_beam_dims(data_scan, axis)

    while len(beam_centers_scan_3d):
        center_scan_3d = beam_centers_scan_3d.pop()
        center_scan_2d = beam_centers_scan_2d.pop()
        beam_dims = beam_dims_scan.pop()
        best_id = -1
        best_dist = -1
        for i, center_gt in enumerate(beam_centers_gt_3d):
            dist = math.dist(center_scan_3d, center_gt)
            if best_id == -1 or dist < best_dist:
                best_id = i
                best_dist = dist
        dist_2d = math.dist(center_scan_2d, beam_centers_gt_2d[best_id])
        if best_dist > 1000:
            continue
        beam_cs_offset_diffs.append(dist_2d)
        beam_length_diffs.append(abs(beam_dims[0] - beam_dims_gt[best_id][0]))
        print("{} | {}".format(beam_dims,beam_dims_gt[best_id]))
        beam_cs_size_diffs.append(abs(beam_dims[1] - beam_dims_gt[best_id][1]) + abs(beam_dims[2] - beam_dims_gt[best_id][2]))
    return beam_cs_offset_diffs, beam_cs_size_diffs, beam_length_diffs


def _get_beam_centers_2d(csv, axis):
    centers = []
    for line in csv:
        parts = line.split(",")
        if parts[0] != "beam" or int(parts[1]) != axis:
            continue
        x_axis = 3 if axis else 2
        center = [float(parts[x_axis]), float(parts[4])]
        centers.append(center)
    return centers


def _get_beam_centers_3d(csv, axis):
    centers = []
    for line in csv:
        parts = line.split(",")
        if parts[0] != "beam" or int(parts[1]) != axis:
            continue
        center = [float(parts[2]), float(parts[3]), float(parts[4])]
        centers.append(center)
    return centers


def _get_beam_dims(csv, axis):
    dims = []
    for line in csv:
        parts = line.split(",")
        if parts[0] != "beam" or int(parts[1]) != axis:
            continue
        dim = [float(parts[5]), float(parts[6]), float(parts[7])]
        if not axis:
            dim[0], dim[1] = dim[1], dim[0]
        dims.append(dim)
    return dims


def _count_types(csv):
    columns = 0
    beams_p = 0
    beams_s = 0

    for line in csv:
        parts = line.split(",")
        if parts[0] == "column":
            columns += 1
        elif parts[0] == "beam":
            if int(parts[1] == 0):
                beams_p += 1
            else:
                beams_s += 1
        else:
            print("Bad type : {}".format(parts[0]))

    return columns, beams_p, beams_s
