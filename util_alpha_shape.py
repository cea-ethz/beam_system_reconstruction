import alphashape
import cv2
import numpy as np

import shapely.geometry

import settings


def analyze_alpha_shape_density(points, density=0.75, name="alpha"):
    alpha_shrinkwrap = alphashape.alphashape(points, 0.005)
    alpha_hull = alphashape.alphashape(points, 0.001)

    area_factor = alpha_shrinkwrap.area / alpha_hull.area

    if settings.read("export.alpha_shape"):
        export_alpha_shapes(points, alpha_shrinkwrap, alpha_hull, name)

    return area_factor > density

def export_alpha_shapes(points, shape_fore, shape_back, name):
    min_x = int(min(shape_fore.bounds[0], shape_back.bounds[0]))
    min_y = int(min(shape_fore.bounds[1], shape_back.bounds[1]))
    max_x = int(max(shape_fore.bounds[2], shape_back.bounds[2]))
    max_y = int(max(shape_fore.bounds[3], shape_back.bounds[3]))

    w = int(max_x - min_x) + 2
    h = int(max_y - min_y) + 2

    img = np.zeros((h + 1, w + 1, 3))
    color_back = (127, 0, 0)
    color_fore = (255, 255, 0)

    offset = np.int32((-min_x, -min_y))

    def _process_points(side):
        _points = np.int32([side.coords]) + offset
        return _points

    if isinstance(shape_back, shapely.geometry.multipolygon.MultiPolygon):
        for polygon in shape_back.geoms:
            cv2.fillPoly(img, _process_points(polygon.exterior), color=color_back)
    else:
        cv2.fillPoly(img, _process_points(shape_back.exterior), color=color_back)

    for point in points:
        img = cv2.circle(img, tuple(np.int32(point) + offset), 3, (0,0,255), 1)

    if isinstance(shape_fore, shapely.geometry.multipolygon.MultiPolygon):
        for polygon in shape_fore.geoms:
            cv2.polylines(img, _process_points(polygon.exterior), color=color_fore, isClosed=True, thickness=3)
            for interior in polygon.interiors:
                print("Interior " + str(interior))
                cv2.polylines(img, _process_points(interior), color=color_fore, isClosed=True, thickness=3)
    else:
        cv2.polylines(img, _process_points(shape_fore.exterior), color=color_fore, isClosed=True, thickness=3)
        for interior in shape_fore.interiors:
            print("Interior " + str(interior))
            cv2.polylines(img, _process_points(interior), color=color_fore, isClosed=True, thickness=3)

    img = cv2.flip(img, 0)
    cv2.imwrite(name, img)


def analyze_alpha_shape_density2(points, density=0.75, name="alpha"):
    alpha_hull = alphashape.alphashape(points, 0.005)

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
        img2[x - 20:x + 20, y - 20:y + 20] = 255

    white_count = np.count_nonzero(img2 == 255)
    density_factor = white_count / alpha_hull.area
    print(density_factor)

    img = cv2.flip(img, 0)
    img2 = cv2.flip(img2, 0)

    img[np.where(img2 == 255)] = (255, 255, 0)

    cv2.imwrite(name, img)

    return density_factor > density





