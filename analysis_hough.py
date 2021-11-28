import cv2
import numpy as np
import skimage.feature
import skimage.transform

import settings
import timer


def analyze_by_hough_transform(pc, aabb):
    scale = 8

    min_bound = aabb.get_min_bound()
    min_bound[0] = int(min_bound[0])
    min_bound[1] = int(min_bound[1])

    accumulator = np.zeros((int(aabb.get_extent()[0]) // scale, int(aabb.get_extent()[1]) // scale))

    points = np.asarray(pc.points)
    for point in points:
        x = int((point[0] - min_bound[0]) // scale)
        y = int((point[1] - min_bound[1]) // scale)

        accumulator[x-5:x+5, y-5:y+5] += 1

    accumulator /= np.max(accumulator)
    accumulator = np.float32(accumulator)
    accumulator *= 255

    ret, accumulator = cv2.threshold(accumulator, 22, 255, cv2.THRESH_BINARY)

    output = cv2.cvtColor(accumulator, cv2.COLOR_GRAY2BGR)

    edges = skimage.feature.canny(accumulator, 2, 1, 25)
    lines = skimage.transform.probabilistic_hough_line(edges, threshold=10, line_length=5, line_gap=3)


    print("{} lines found".format(len(lines)))
    for line in lines:
        p0, p1 = line
        cv2.line(output, p0, p1, (255, 0, 0))

    cv2.imwrite("map.png", accumulator)

    output = output.astype(np.uint8)
    cv2.imwrite("hough.png", output)

    if settings.read("display.hough"):
        timer.pause()
        cv2.imshow("hough", output)
        cv2.waitKey()
        cv2.destroyAllWindows()
        timer.unpause()
