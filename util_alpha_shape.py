import alphashape
import cv2
import numpy as np
import shapely.geometry

import settings
import timer
import ui


def analyze_alpha_shape_density2(points, density=0.75, name="alpha"):
    """
    Analyzes the 'density' of a point cloud, to discern a cloud with smoothly spread out points vs locally clustered
    points, regardless of total point count. This is to discern beam grids from floors and walls.

    :param points:
    :param density:
    :param name:
    :return:
    """

    #timer.start("alpha")

    alpha_hull = alphashape.alphashape(points, 0)

    w = int(alpha_hull.bounds[2] - alpha_hull.bounds[0]) + 2
    h = int(alpha_hull.bounds[3] - alpha_hull.bounds[1]) + 2

    img = np.zeros((h + 1, w + 1, 3))
    img2 = np.zeros((h+1, w+1))
    color_back = (127, 0, 0)
    color_fore = (255, 255, 0)

    offset = np.int32((-alpha_hull.bounds[0], -alpha_hull.bounds[1]))

    def _process_points(side):
        _points = np.int32([side.coords]) + offset
        return _points

    if isinstance(alpha_hull, shapely.geometry.multipolygon.MultiPolygon):
        for polygon in alpha_hull.geoms:
            cv2.fillPoly(img, _process_points(polygon.exterior), color=color_back)
    else:
        cv2.fillPoly(img, _process_points(alpha_hull.exterior), color=color_back)

    points = np.int32(points) + offset

    for point in points:
        x = point[1]
        y = point[0]
        of = settings.read("tuning.alpha_density_point_size")
        img2[x - of:x + of, y - of:y + of] = 255

    white_count = np.count_nonzero(img2 == 255)

    density_factor = white_count / alpha_hull.area

    print(" " + name + " : " + str(round(density_factor, 3)))

    img = cv2.flip(img, 0)
    img2 = cv2.flip(img2, 0)

    img[np.where(img2 == 255)] = (255, 255, 0)

    cv2.imwrite(ui.dir_output + "alpha_shapes/" + name, img)

    #timer.end("alpha")

    return density_factor > density





