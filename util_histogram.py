"""
Utilities for processing histogram data
"""

import numpy as np


def smooth_histogram(hist, extension=1):
    """Apply simple smoothing by setting each each histogram value to the total of its neighborhood"""
    # Ugly for now but meh, doesn't account for first and last positions
    hist_copy = np.copy(hist)
    for i in range(len(hist)):
        hist[i] = 0
        for j in range(-extension, extension + 1):
            new_i = i + j
            if new_i < 0 or new_i >= len(hist):
                continue
            hist[i] += hist_copy[new_i]
    return hist


def normalize_histogram(hist):
    """Normalize histogram values to 0-1 scale based on largest value"""
    hist = hist.astype('float')
    hist /= np.max(hist)
    return hist


def process_histogram(hist):
    """Apply smoothing and normalization"""
    hist_smooth = smooth_histogram(np.copy(hist), extension=2)
    hist_smooth = normalize_histogram(hist_smooth)
    hist = normalize_histogram(hist)
    return hist, hist_smooth

def get_peak_slice_params(hist, peak, diff=0.1):
    """
    Given a peak, return the slice parameters to stay within [diff] of that peak (may not be centered at the peak)
    :param hist:
    :param peak:
    :param diff:
    :return:
    """
    low = peak
    high = peak+1

    for i in range(low - 1, -1, -1):
        if hist[peak] - hist[i] < diff:
            low = i
        else:
            break
    for i in range(low + 1, len(hist)):
        if hist[peak] - hist[i] < diff:
            high = i
        else:
            break

    width = high - low
    position = width / 2 + low
    return position, width
