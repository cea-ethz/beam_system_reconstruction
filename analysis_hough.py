import cv2
import numpy as np
import skimage.feature
import skimage.transform

from matplotlib import cm

import timer


def analyze_by_hough_transform(pc, aabb):
    scale = 4

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
    output = cv2.cvtColor(accumulator, cv2.COLOR_GRAY2BGR)


    accumulator *= 255
    accumulator = accumulator.astype(np.int32)

    edges = skimage.feature.canny(accumulator, 2, 1, 25)
    lines = skimage.transform.probabilistic_hough_line(edges, threshold=10, line_length=5, line_gap=3)


    print("{} lines found".format(len(lines)))
    for line in lines:
        p0, p1 = line
        cv2.line(output, p0, p1, (255, 0, 0))
    #print(output)
    #output *= 255
    cv2.imwrite("hough.png", output)

    #accumulator *= 255
    cv2.imwrite("map.png", accumulator)

    timer.pause()
    cv2.imshow("hough", output)
    cv2.waitKey()
    cv2.destroyAllWindows()
    timer.unpause()
