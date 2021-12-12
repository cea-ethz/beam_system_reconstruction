import math


def check_element_counts(data_gt, data_scan):
    columns_gt, beams_p_gt, beams_s_gt = _count_types(data_gt)
    columns_scan, beams_p_scan, beams_s_scan = _count_types(data_scan)

    return columns_scan - columns_gt, (beams_p_scan - beams_p_gt) + (beams_s_scan - beams_s_gt)


def check_cross_section_offset(data_gt, data_scan):
    column_diffs = []
    column_centers_gt = get_column_centers(data_gt)
    column_centers_scan = get_column_centers(data_scan)

    while len(column_centers_scan):
        center_scan = column_centers_scan.pop()
        best_id = -1
        best_dist = -1
        for i, center_gt in enumerate(column_centers_gt):
            dist = math.dist(center_scan, center_gt)
            if best_id == -1 or dist < best_dist:
                best_id = i
                best_dist = dist
        column_diffs.append(best_dist)
        del column_centers_gt[best_id]

    #print(column_diffs)
    return sum(column_diffs) / len(column_diffs)


def get_column_centers(csv):
    centers = []
    for line in csv:
        print(line)
        parts = line.split(",")
        if parts[0] != "column":
            continue
        center = [float(parts[2]), float(parts[3])]
        centers.append(center)
    return centers


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
