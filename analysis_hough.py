import cv2
import numpy as np
import skimage.feature
import skimage.transform

import settings
import timer
import ui


class Segment():
    def __init__(self):
        pass

class Strand(Segment):
    def __init__(self):
        pass


def cloud_to_accumulator(points, aabb,scale=8):
    min_bound = aabb.get_min_bound()
    min_bound[0] = int(min_bound[0])
    min_bound[1] = int(min_bound[1])

    accumulator = np.zeros((int(aabb.get_extent()[0]) // scale, int(aabb.get_extent()[1]) // scale))

    for point in points:
        x = int((point[0] - min_bound[0]) // scale)
        y = int((point[1] - min_bound[1]) // scale)

        accumulator[x-5:x+5, y-5:y+5] += 1

    accumulator /= np.max(accumulator)
    accumulator = np.float32(accumulator)
    accumulator *= 255

    return accumulator


def analyze_by_hough_transform(pc, aabb):
    timer.start("Hough Analysis")

    accumulator = cloud_to_accumulator(np.array(pc.points), aabb)

    cv2.imwrite(ui.dir_output + "accumulator.png", accumulator)

    ret, accumulator = cv2.threshold(accumulator, 22, 255, cv2.THRESH_BINARY)

    output = cv2.cvtColor(accumulator, cv2.COLOR_GRAY2BGR)



    edges = skimage.feature.canny(accumulator, 2, 1, 25)
    lines = skimage.transform.probabilistic_hough_line(edges, threshold=10, line_length=5, line_gap=3)

    print("{} lines found".format(len(lines)))
    for line in lines:
        p0, p1 = line
        cv2.line(output, p0, p1, (255, 0, 0))

    for line in lines:
        p0, p1 = line
        output[p0[1], p0[0]] = (0, 0, 255)
        output[p1[1], p0[0]] = (0, 0, 255)

    cv2.imwrite(ui.dir_output + "accumulator_threshold.png", accumulator)

    output = output.astype(np.uint8)
    cv2.imwrite(ui.dir_output + "hough.png", output)

    if settings.read("display.hough"):
        timer.pause("Hough Analysis")
        cv2.imshow("hough", output)
        cv2.waitKey()
        cv2.destroyAllWindows()
        timer.unpause("Hough Analysis")

    timer.end("Hough Analysis")
