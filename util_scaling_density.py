import cv2
import numpy as np

import timer
import ui


def compute_scaling_density(points, name):
    #timer.start("density")

    h = np.max(points[:, 0]) - np.min(points[:, 0])
    w = np.max(points[:, 1]) - np.min(points[:, 1])

    img_shape = np.array((w + 1, h+1)).astype(int)

    offset = np.int32((np.min(points[:, 0]), np.min(points[:, 1])))
    points -= offset

    points = points.astype(np.int32)

    points[:, 1] = img_shape[0] - points[:, 1]

    factor_a = 40
    factor_b = 200

    scale_a = _get_level(points, factor_a, img_shape)
    count_a = np.count_nonzero(scale_a == 255) * factor_a * factor_a
    cv2.imwrite(ui.dir_output + "scaling_density/" + name + "_a.png", scale_a)

    scale_b = _get_level(points, factor_b, img_shape)
    count_b = np.count_nonzero(scale_b == 255) * factor_b * factor_b
    cv2.imwrite(ui.dir_output + "scaling_density/" + name + "_b.png", scale_b)

    #timer.end("density")

    print(f"Density : {count_a / count_b}")

    return count_a / count_b


def _get_level(points, factor, img_shape):
    points = np.copy(points)
    points = points / factor
    points = points.astype(int)

    img_scaled = np.zeros(img_shape // factor + np.array((1, 1)))

    for point in points:
        img_scaled[point[1], point[0]] = 255

    return img_scaled

