"""
Utilities for processing histogram data
"""

import numpy as np

color_back = 'gray'
color_back_highlight = 'cyan'
color_front = 'C0'
color_front_highlight = 'cyan'


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
    low = peak-1
    high = peak+1

    for i in range(low - 1, -1, -1):
        if hist[peak] - hist[i] < diff:
            low = i
        else:
            break
    for i in range(low+1, len(hist)): # TK should this be high instead of low+1? Borks things when tried but maybe...
        if hist[peak] - hist[i] < diff:
            high = i
        else:
            break

    width = high - low
    position = width / 2 + low
    return position, width


def render_bar(ax, hist_a, hist_b, peaks):
    bar_list_b = ax.bar(range(len(hist_b)), hist_b, color=color_back, width=1)
    bar_list_a = ax.bar(range(len(hist_a)), hist_a, color=color_front, width=1)
    mean_b = np.mean(hist_b)
    ax.axhline(mean_b, color='orange')

    for peak in peaks:
        bar_list_b[peak].set_color(color_back_highlight)
        bar_list_a[peak].set_color(color_front_highlight)
