#!/usr/bin/env python
# coding: utf-8
############################################################################################################
# Authors:
#   Jack Monahan,       EMBL-EBI                                                    (First prototype)
#   Tomas Fitzgerald,   EMBL-EBI                                                    (Fast mode)
#   Marcio Ferreira,    EMBL-EBI,       marcio@ebi.ac.uk                            (Current Maintainer)
#   Sebastian Stricker, Uni Heidelberg, sebastian.stricker@stud.uni-heidelberg.de   (Current Maintainer)
# Date: 08/2021
# License: Contact authors
###
# Algorithms for:
#   cropping medaka embryo videos from the Acquifier Imaging machine
#   detecting heart in medaka embryo videos,
#   calculating bpm frequency from pixel color fluctuation in said videos
###
############################################################################################################
from collections import Counter, OrderedDict
import warnings
import multiprocessing
from joblib import Parallel, delayed
import seaborn as sns
from matplotlib import pyplot as plt
import operator
import statistics
import pandas as pd
import os
import glob2
import random
import logging


from statistics import mean

import numpy as np
import cv2

# import skimage
from skimage.util import img_as_ubyte, img_as_float
from skimage.filters import threshold_triangle, threshold_yen
from skimage.measure import label
# from skimage.metrics import structural_similarity
from skimage import color, feature

# import scipy
from scipy.stats import gaussian_kde
from scipy import signal
from scipy.signal import find_peaks, savgol_filter  # , peak_prominences, welch
from scipy.interpolate import CubicSpline

import matplotlib
matplotlib.use('Agg')

# Parallelisation
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=SyntaxWarning)

################################################################################
##########################
##  Globals   ##
##########################

# Improve contrast with CLAHE (Contrast Limited Adaptive Histogram Equalization)
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html#histogram-equalization
# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

LOGGER = logging.getLogger(__name__)

# Kernel for image smoothing
kernel = np.ones((5, 5), np.uint8)
################################################################################
##########################
##  Functions   ##
##########################
# start of fast mode


def get_image_files_and_time_spacing(indir, frame_format, well_number, loop):

    if frame_format == "tiff":
        well_frames = glob2.glob(indir + '/*' + well_number + '*.tif') + \
            glob2.glob(indir + '/*' + well_number + '*.tiff')
        well_frames = [fname for fname in well_frames if loop in fname]
    if frame_format == "jpg":
        well_frames = glob2.glob(indir + '/*' + well_number + '*.jpeg') + \
            glob2.glob(indir + '/*' + well_number + '*.jpg')
        well_frames = [fname for fname in well_frames if loop in fname]
    best_dim = {}
    for i in range(0, len(well_frames)):
        if not cv2.imread(well_frames[i], 1) is None:
            if cv2.imread(well_frames[i], 1).shape not in best_dim:
                best_dim[cv2.imread(well_frames[i], 1).shape] = 0
            else:
                best_dim[cv2.imread(well_frames[i], 1).shape] = best_dim[cv2.imread(
                    well_frames[i], 1).shape] + 1
    dimension = max(best_dim.items(), key=operator.itemgetter(1))[0]
    times = []
    valid_frames = []
    for i in range(0, len(well_frames)):
        if not cv2.imread(well_frames[i], 1) is None:
            if cv2.imread(well_frames[i], 1).shape == dimension:
                valid_frames.append(well_frames[i])
                time = float([s for s in well_frames[i].split(
                    "--") if s.startswith('T') and not s.startswith('TM')][0][1:])
                times.append(time)
    well_frames = [x for _, x in sorted(zip(times, valid_frames))]
    sorted_times = [x for _, x in sorted(zip(times, times))]
    timestamp0 = sorted_times[0]
    timestamp_final = sorted_times[-1]
    total_time = (timestamp_final - timestamp0) / 1000
    time_d = []
    for i in range(0, len(sorted_times)):
        time_d.append((sorted_times[i] - timestamp0) / 1000)
    return {"time": time_d, "files": well_frames, "fps": len(time_d)/total_time}

# Use a method to restrict differences to only no changes between frames - with each non change adding 1 to the final mask
# NB. this means that points that move in all frames will have an intensity of 0 in the mask filter
# NB. And e.g. if we want to include  points that move in 10% of frames low_bound would equal len(well_frame)*0.1
# NB. here we also apply some blurring and then the standard lower bound inclusion - there are the only two parameters and could could be dynamically devired


def define_mask_and_extract_pixels(well_frames, blur_par=11, low_bound=10):
    img = cv2.absdiff(cv2.imread(
        well_frames[0], 1), cv2.imread(well_frames[0], 1))
    for i in range(1, len(well_frames)):
        c1 = cv2.imread(well_frames[i-1], 1)
        c2 = cv2.imread(well_frames[i], 1)
        if c1.shape == c2.shape:
            img1 = cv2.absdiff(c1, c2)
            img1 = img1 == 0
            img = img+img1
    img2 = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    img3 = cv2.medianBlur(img2, blur_par)
    ret, mask = cv2.threshold(img3, low_bound, 255, cv2.THRESH_BINARY)
    mask_pixels_and_medians = {}
    mask_pixels_and_medians['all'] = []
    mask_pixels_and_medians['mask'] = mask
    mask_pixels_and_medians['medians'] = []
    for i in range(0, len(well_frames)):
        img = cv2.imread(well_frames[i], 1)
        masked_data = cv2.bitwise_and(img, img, mask=mask)
        pixels = img[mask < 255, 0]
        mask_pixels_and_medians['all'].append(pixels)
        mask_pixels_and_medians['medians'].append(np.median(pixels))
    return mask_pixels_and_medians

# FFT on median pixel values across frames - with a manual power spectrum calculation
# NB. transform frequency estimate at maximum power spectra to bpm


def estimate_bpm_fft(frame_pixel_medians, time):
    data = np.array(frame_pixel_medians)
    fourier_transform = np.fft.fft(data)
    I = np.abs(fourier_transform)**2/len(fourier_transform)
    N = int(len(I)/2)
    P = (4/len(I))*I[0:N]
    timestep = np.mean(np.diff(time))
    F = np.fft.fftfreq(len(I), d=timestep)[0:N]
    result = list(sorted(zip(np.delete(P, 0), np.delete(F, 0)), reverse=True))
    result = list(filter(lambda p: p[1] > 0.5 and p[1] < 5, result))
    freq_at_max_peak = result[0][1]
    bpm = freq_at_max_peak * 60
    return {"F": F, "P": P, "bpm": bpm}

# Wrapper to run the methods for a single 'well' position and 'loop' number


def run_a_well(indir, out_dir, frame_format, well_number, loop, blur_par=11, low_bound=10):

    files_and_time = get_image_files_and_time_spacing(
        indir, frame_format, well_number, loop)

    pixels = define_mask_and_extract_pixels(
        files_and_time['files'], blur_par, low_bound)
    freq_power_bpm = estimate_bpm_fft(
        pixels['medians'], files_and_time['time'])
    return freq_power_bpm
# end of fast mode
###############################################################
# Detect embryo in image based on Hough Circle Detection


def detectEmbryo(frame):
    # Find circle i.e. the embryo in the yolk sac
    img_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur
    img_grey = cv2.GaussianBlur(img_grey, (9, 9), 0)

    # Edge detection
    edges = feature.canny(img_as_float(img_grey), sigma=3)
    edges = img_as_ubyte(edges)

#   edges = cv2.Canny(img_grey, 100, 200)

    # Circle detection
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1,
                               150, param1=50, param2=30, minRadius=150, maxRadius=400)

    # If fails to detect embryo following edge detection,
    # try with the original image
    if circles is None:
        # print("fails detect circle with edges in detect_embryo()")
        circles = cv2.HoughCircles(img_grey, cv2.HOUGH_GRADIENT, 1,
                                   150, param1=50, param2=30, minRadius=150, maxRadius=400)

    #################################################################################
    if circles is None:  # new feature

        # Both trials failed to detect embryo, then try to threshold the image first, and try again

        plt.imshow(img_grey)
        plt.show()

        ret, th = cv2.threshold(
            img_grey, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        plt.imshow(th)
        plt.show()
        circles = cv2.HoughCircles(
            th, cv2.HOUGH_GRADIENT, 1, 150, param1=50, param2=30, minRadius=150, maxRadius=400)

    #################################################################################

    if circles is not None:
        # Sort detected circles
        circles = sorted(circles[0], key=lambda x: x[2], reverse=True)
        # Only take largest circle to be embryo
        circle = np.uint16(np.around(circles[0]))

        # Circle coords
        centre_x = circle[0]
        centre_y = circle[1]
        radius = circle[2]
        x1 = centre_x - radius
        x2 = centre_x + radius
        y1 = centre_y - radius
        y2 = centre_y + radius

        # Round coords
        x1_test = 100 * round(x1 / 100)
        x2_test = 100 * round(x2 / 100)
        y1_test = 100 * round(y1 / 100)
        y2_test = 100 * round(y2 / 100)

        # If rounded figures are greater than x1 or y1, take 50 off it
        if x1_test > x1:
            x1 = x1_test - 50
        else:
            x1 = x1_test

        if y1_test > y1:
            y1 = y1_test - 50
        else:
            y1 = y1_test

        # If rounded figures are less than x2 or y2, add 50
        if x2_test < x2:
            x2 = x2_test + 50
        else:
            x2 = x2_test

        if y2_test < y2:
            y2 = y2_test + 50
        else:
            y2 = y2_test

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

    else:
        # print('Fails all trials to detect embryo in detect_embryo()')
        circle = None
        x1 = None
        y1 = None
        x2 = None
        y2 = None

    return(circle, x1, y1, x2, y2)

# ## Function greyFrames(frames)
# Convert all frames in a list into greyscale


def greyFrames(frames, stop_frame=0):

    grey_frames = []
    frame_number = 0
    for frame in frames:

        # Check that frame exists
        if (frame is not None) and (frame.size > 0):

            # Convert RGB to greyscale
            grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        else:
            grey_frame = frame

        grey_frames.append(grey_frame)
        frame_number += 1

        # if stop_frame in greater than ), it means that the embrio flipped around, then we need to skip frames beyond the frame which the embryo mopves
        if (stop_frame > 0) and (frame_number >= stop_frame):
            break

    return(grey_frames)

# ## Function resizeFrames(frames, scale = 50)
# Uniformly resize frames based on common scaling factor e.g. 50, will halve size


def resizeFrames(frames, scale=50):

    resized_frames = []
    for frame in frames:

        # Check that frame exists
        if (frame is not None) and (frame.size > 0):

            width = int(frame.shape[1] * scale / 100)
            height = int(frame.shape[0] * scale / 100)
            dim = (width, height)

            # Resize frame based on intrpolated pixels values
            # Increase frame size
            if scale > 100:
                resized = cv2.resize(
                    frame, dim, interpolation=cv2.INTER_LINEAR)
            # Reduce frame size
            else:
                resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        else:
            resized = None

        resized_frames.append(resized)

    return(resized_frames)

# ## Function normVideo(frames)
# Normalise across frames to harmonise intensities (& possibly remove flickering)
# TODO: Streching should be done from the bottom as well.
# TODO: Also, a single outlier will worsen the normalization
# TODO: Could be done more efficient probably.


def normVideo(frames):

    # Start at first non-empty frame
    #   first_frame = next(x for x, frame in enumerate(frames) if frame is not None)
    #   for i in range(first_frame, len(frames)):
    for i in range(len(frames)):

        frame = frames[i]

        if (frame is not None) and (frame.size > 0):

            # Convert RGB to greyscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Add to frame stack
            try:
                filtered_frames = np.dstack((filtered_frames, frame))
            # Initiate as array if doesn't exist
            except UnboundLocalError:
                filtered_frames = np.asarray(frame)

    norm_frames = []

    max_in_frames = np.max(filtered_frames)
    # Divide by max to try and remove flickering between frames
    for i in range(len(frames)):

        frame = frames[i]

        if (frame is not None) and (frame.size > 0):

            # Convert RGB to greyscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            norm_frame = np.uint8(frame / max_in_frames * 255)

            # Convert scaled greyscale back to RGB
            norm_frame = cv2.cvtColor(norm_frame, cv2.COLOR_GRAY2BGR)

        # If empty frame
        else:
            norm_frame = None

        norm_frames.append(norm_frame)

    return(norm_frames)

# ## Function processFrame(frame)
# Pre-process frame


def processFrame(frame):
    """Image pre-processing and illumination normalisation"""

    # Convert RGB to LAB colour
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # Split the LAB image into different channels
    l, a, b = cv2.split(lab)

    # Improve contrast with CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Apply CLAHE to L-channel
    # TODO: Clahe gives great contrast but increases noise between frames dramatically.
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L-channel with the A and B channel
    limg = cv2.merge((cl, a, b))

    # Convert to greyscale
    frame_grey = cl

    # Convert CLAHE-normalised greyscale frame back to BGR
    out_frame = cv2.cvtColor(frame_grey, cv2.COLOR_GRAY2BGR)

    # Blur the CLAHE frame
    # Blurring kernel numbers must be odd integers
    blurred_frame = cv2.GaussianBlur(frame_grey, (9, 9), 0)

    # this is an option, and must be tested.
    # blurred_frame = cv2.bilateralFilter(frame_grey, 7, 50, 50)

    return out_frame, frame_grey, blurred_frame

# ## Function maskFrame(frame, mask)


def maskFrame(frame, mask):
    """Add constant value in green channel to region of frame from the mask."""

    # split source frame into B,G,R channels
    b, g, r = cv2.split(frame)

    # add a constant to G (green) channel to highlight the masked area
    g = cv2.add(g, 50, dst=g, mask=mask, dtype=cv2.CV_8U)
    masked_frame = cv2.merge((b, g, r))

    return masked_frame

# ## Function filterMask(mask, min_area = 300):


def filterMask(mask, min_area=300):

    # Contour mask
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Filter contours based on their area
    filtered_contours = []
    for i in range(len(contours)):
        contour = contours[i]

        # Filter contours by their area
        if cv2.contourArea(contour) >= min_area:
            filtered_contours.append(contour)

    contours = filtered_contours

    # Create blank mask
    rows, cols = mask.shape
    filtered_mask = np.zeros(shape=[rows, cols], dtype=np.uint8)

    # Draw and fill-in filtered contours on blank mask
    cv2.drawContours(filtered_mask, contours, -1, 255, thickness=-1)

    return filtered_mask

# ## Function diffFrame(frame, frame2_blur, frame1_blur, min_area = 300):
# Differences between two frames


def diffFrame(frame, frame2_blur, frame1_blur, min_area=300):
    """Calculate the abs diff between 2 frames and returns frame2 masked with the filtered differences."""

    # Image Structural similarity
#   _ , diff = structural_similarity(frame2_blur, frame1_blur, full = True)
    # invert colour
#   diff = diff * -1

    # Absolute difference between frames
    diff = cv2.absdiff(frame2_blur, frame1_blur)

    # Make sure there are differences...
#   sum_diff = np.sum(diff)
#   if sum_diff > 0:
    try:

        # Triangle thresholding on differences
        triangle = threshold_triangle(diff)
        thresh = diff > triangle
        thresh = thresh.astype(np.uint8)

        # Opening to remove noise
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours in mask and filter them based on their area
        mask = filterMask(mask=thresh, min_area=min_area)

        # Mask frame
        masked_frame = maskFrame(frame, mask)

    except ValueError:
        masked_frame = frame
        thresh = diff

    # Return the masked frame, the filtered mask and the absolute differences for the 2 frames
    # return masked_frame, mask, thresh
    return masked_frame, thresh

# ## Function rolling_diff(index, frames, win_size = 5, direction = "forward", min_area = 300):
# Forward or reverse rolling window of width w with step size ws


def rolling_diff(index, frames, win_size=5, direction="forward", min_area=300):
    """
    Implement rolling window
    * win_size INT
        Window size (default = 5)

           """

    if direction == "forward":

        if (index + win_size) > len(frames):
            window_indices = list(range(index, len(frames)))
        else:
            window_indices = list(range(index, index + win_size))

        frame0 = frames[window_indices[0]]
        _, _, old_blur = processFrame(frame0)

        # Determine absolute differences between current and previous frame
        # Frame[j-1] vs. frame[j]...

        # Generate blank images for masking
        rows, cols, _ = frame0.shape
        abs_diffs = np.zeros(shape=[rows, cols], dtype=np.uint8)

        for i in window_indices[1:]:

            frame = frames[i]

            if frame is not None:
                _, _, frame_blur = processFrame(frame)
                _, triangle_thresh = diffFrame(frame, frame_blur, old_blur)
                abs_diffs = cv2.add(abs_diffs, triangle_thresh)

    elif direction == "reverse":

        if index >= win_size:
            window_indices = list(range(index, index - win_size, -1))[::-1]
        else:
            window_indices = list(range(0, index + 1))

        frame = frames[window_indices[-1]]
        _, _, frame_blur = processFrame(frame)

        # Determine absolute differences between current and previous frame
        # Frame[i] vs frame[i - 1] .... [(i - 3]

        # Generate blank images for masking
        rows, cols, _ = frame.shape
        abs_diffs = np.zeros(shape=[rows, cols], dtype=np.uint8)

        for i in window_indices[:-1]:

            frame0 = frames[i]

            if frame0 is not None:

                _, _, old_blur = processFrame(frame0)

                _, triangle_thresh = diffFrame(frame, frame_blur, old_blur)
                abs_diffs = cv2.add(abs_diffs, triangle_thresh)

                # if the embryo flip around, then the sum of mask is high and scape the following frames
                sum_of_mask = np.sum(abs_diffs)

                if sum_of_mask > 50000:
                    movement = True
                    break
                else:
                    movement = False

    # Filter mask by area
    # Opening to remove noise
    # thresh = cv2.morphologyEx(abs_diffs, cv2.MORPH_OPEN, kernel)

    # # TODO: Is necessary? Romoves a lot of data potentially
    thresh = cv2.erode(abs_diffs, (7, 7), iterations=3)

    # Filter based on their area
    thresh = filterMask(mask=thresh, min_area=min_area)

    # Mask frame
    masked_frame = maskFrame(frame, thresh)

    return(masked_frame, abs_diffs, thresh, movement)

# ## Function heartQC_plot(ax, f0_grey, heart_roi,heart_roi_clean,label_maxima, figsize=(15, 15))
# Generate figure highlighting the probable heart region


def heartQC_plot(ax, f0_grey, heart_roi, heart_roi_clean, overlay):
    """
    Generate figure to QC heart segmentation
    """

    # First frame
    ax[0, 0].imshow(f0_grey, cmap='gray')
    ax[0, 0].set_title('Embryo', fontsize=10)
    ax[0, 0].axis('off')

    # Summed Absolute Difference between sequential frames
    ax[0, 1].imshow(heart_roi)
    ax[0, 1].set_title('Summed Absolute\nDifferences', fontsize=10)
    ax[0, 1].axis('off')

    # Thresholded Differences
    ax[1, 0].imshow(heart_roi_clean)
    ax[1, 0].set_title('Thresholded Absolute\nDifferences', fontsize=10)
    ax[1, 0].axis('off')

    # Overlap between filtered RoI mask and pixel maxima
    ax[1, 1].imshow(overlay)
    ax[1, 1].set_title('RoI overlap with maxima', fontsize=10)
    ax[1, 1].axis('off')

    return(ax)

# ## Function qc_mask_contours(heart_roi)


def new_qc_mask_contours(heart_roi, maxima):

 # Find contours based on thresholded, summed absolute differences
    contours, _ = cv2.findContours(
        heart_roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Filter contours based on which overlaps with the most changeable pixels
    rows, cols = heart_roi.shape
    final_mask = np.zeros(shape=[rows, cols], dtype=np.uint8)
    img = np.zeros(shape=[rows, cols], dtype=np.uint8)
    regions = 0
    pRatio = []
    pOverlap = []
    iList = []

    for i in range(len(contours)):

        # Contour to test
        test_contour = contours[i]

        # Create blank mask
        contour_mask = np.zeros(shape=[rows, cols], dtype=np.uint8)

        # Calculate overlap with the most changeable pixels i.e. max_opening
        # Draw and fill-in filtered contours on blank mask (contour has to be in a list)
        cv2.drawContours(contour_mask, [test_contour], -1, (255), thickness=-1)

        contour_pixels = (contour_mask / 255).sum()

        # Find overlap between contoured area and the N most changeable pixels
        overlap = np.logical_and(maxima, contour_mask)
        overlap_pixels = overlap.sum()

        pOverlap.append(overlap_pixels)
        # print(overlap_pixels)
        # print(contour_pixels)
        # Calculate ratio between area of intersection and contour area
        pixel_ratio = overlap_pixels / contour_pixels

        pRatio.append(pixel_ratio)

        iList.append(i)

        # Take all regions that overlap with 80%+ of the 250 most changeable pixels or
        # those in which these pixels comprise at least 5% of the pixels in the contoured region
        # if (overlap_pixels >= (top_pixels * 0.8)) or (pixel_ratio >= 0.05):

        # final_mask = cv2.add(final_mask, contour_mask)
        # regions += 1

    max_value_rate = max(pRatio)
    max_value_pixels = max(pOverlap)
    indexOfMaximum_r = [pRatio.index(max_value_rate)]
    indexOfMaximum_p = [pOverlap.index(max_value_pixels)]

    if indexOfMaximum_r != indexOfMaximum_p:
        final_list = indexOfMaximum_r + indexOfMaximum_p
        regions = 2
    else:
        final_list = indexOfMaximum_r
        regions = 1

    new_countours = [contours[i] for i in final_list]

    contour_mask = np.zeros(shape=[rows, cols], dtype=np.uint8)

    cv2.drawContours(contour_mask, new_countours, -1, (255), thickness=-1)

    final_mask = cv2.add(final_mask, contour_mask)

    return(final_mask, regions)


def qc_mask_contours(heart_roi, maxima, top_pixels):
    # Find contours based on thresholded, summed absolute differences
    contours, _ = cv2.findContours(
        heart_roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Filter contours based on which overlaps with the most changeable pixels
    rows, cols = heart_roi.shape
    final_mask = np.zeros(shape=[rows, cols], dtype=np.uint8)
    img = np.zeros(shape=[rows, cols], dtype=np.uint8)
    regions = 0

    for i in range(len(contours)):

        # Contour to test
        test_contour = contours[i]

        # Create blank mask
        contour_mask = np.zeros(shape=[rows, cols], dtype=np.uint8)

        # Calculate overlap with the most changeable pixels i.e. max_opening
        # Draw and fill-in filtered contours on blank mask (contour has to be in a list)
        cv2.drawContours(contour_mask, [test_contour], -1, (255), thickness=-1)

        contour_pixels = (contour_mask / 255).sum()

        # Find overlap between contoured area and the N most changeable pixels
        overlap = np.logical_and(maxima, contour_mask)
        overlap_pixels = overlap.sum()

        # Calculate ratio between area of intersection and contour area
        pixel_ratio = overlap_pixels / contour_pixels

        # Take all regions that overlap with 80%+ of the 250 most changeable pixels or
        # those in which these pixels comprise at least 5% of the pixels in the contoured region
        if (overlap_pixels >= (top_pixels * 0.8)) or (pixel_ratio >= 0.05):
            final_mask = cv2.add(final_mask, contour_mask)
            regions += 1

    return(final_mask, regions)

# ## Function qc_mask_contours(heart_roi)


def new2_qc_mask_contours(heart_roi):
    area_count = []
    rows, cols = heart_roi.shape
    contour_mask = np.zeros(shape=[rows, cols], dtype=np.uint8)

    # Find contours based on thresholded, summed absolute differences
    contours, _ = cv2.findContours(
        heart_roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Filter contours based on which overlaps with the most changeable pixels
    rows, cols = heart_roi.shape
    final_mask = np.zeros(shape=[rows, cols], dtype=np.uint8)
    img = np.zeros(shape=[rows, cols], dtype=np.uint8)
    regions = 0

    for i in range(len(contours)):

        # Contour to measure area
        test_contour = contours[i]
        area = cv2.contourArea(test_contour)
        area_count.append(area)
        regions += 1

    max_value = max(area_count)
    max_index = area_count.index(max_value)
    cv2.drawContours(
        contour_mask, [contours[max_index]], -1, (255), thickness=-1)
    final_mask = cv2.add(final_mask, contour_mask)

    return(final_mask, regions)

# Detect extreme outliers and filter them out
# Outliers due to e.g. sudden movement of whole embryo or flickering light


def iqrFilter(times, signal):

    # Calculate interquartile range from signal
    q3, q1 = np.percentile(signal, [75, 25])
    iqr = q3 - q1

    # Filter signal based on IQR
    iqr_upper = q3 + (iqr * 1.5)
    iqr_lower = q1 - (iqr * 1.5)

    passed = (signal > iqr_lower) & (signal < iqr_upper)
    filtered_times = times[passed]
    filtered_signal = signal[passed]

    return(filtered_times, filtered_signal)

# ## Function iqrFilter(times, signal)
# Interpolate pixel or regional signal, filtering if necessary


def interpolate_signal(times, y, empty_frames):

    # No filtering needed for interpolation if no empty frames
    if len(empty_frames) == 0:

        # Remove any linear trends
        detrended = signal.detrend(y)

        # print('after detrend')
        # print(len(times))
        # print(len(detrended))

        # Filter signal with IQR
        times_final, y_final = iqrFilter(times, detrended)

    # Filter out missing signal values before interpolation
    else:

        # Remove NaN values from signal and time domain for interpolation
        y_filtered = y.copy()
        y_filtered = np.delete(y_filtered, empty_frames)
        times_filtered = times.copy()
        times_filtered = np.delete(times_filtered, empty_frames)

        # Remove any linear trends
        detrended = signal.detrend(y_filtered)

        # Filter signal with IQR
        times_final, y_final = iqrFilter(times_filtered, detrended)

    # Perform cubic spline interpolation to calculate in missing values
    cs = CubicSpline(times_final, y_final)

    return(times_final, y_final, cs)

# ## Function interpolate_signal(times, y, empty_frames)
# Detrend heart signal and smoothe


def detrendSignal(interpolated_signal, time_domain, window_size=15):
    """
    Use Savitzky–Golay convolution filter to smoothe time-series data.
    Fits successive sub-sets of adjacent data points with a low-degree polynomial using linear least squares.
    Increases the precision of the data without distorting the signal tendency.
    Sav-Gol filter AKA LOESS (locally estimated scatterplot smoothing)
    * interpolated_signal
        Interpolated raw signal intensities.
    * time_domain
        Time domain generated from data interpolation.
    * window_size INT
        Odd number specifying window size for Savitzky–Golay filter.
    """
    # Savitzky–Golay filter to smooth signal
    # Window size must be odd integer
    data_detrended = savgol_filter(
        interpolated_signal(time_domain), window_size, 3)

    # Normalise Signal
    normalised_signal = data_detrended / data_detrended.std()

    # Generate new Cubic Spline based on smoothed data
    norm_cs = CubicSpline(time_domain, normalised_signal)

    return(norm_cs)

# ## Function MAD(numeric_vector)
# Calculate Median Absolute Deviation
# for a given numeric vector


def MAD(numeric_vector):

    median = np.median(numeric_vector)
    diff = abs(numeric_vector - median)
    med_abs_deviation = np.median(diff)

    return med_abs_deviation

# ## Function def rolling_window(signal, win_size = 20, win_step = 5, direction = "forward")
# Forward or reverse rolling window W with step size Ws


def rolling_window(signal, win_size=20, win_step=5, direction="forward"):
    """
    Implement rlling window over Nanopore read signal intensities (mean or median)
    * signal FLOAT
         numeric vector
    * win_size INT
        Window size (default = 20)
    * win_step INT
        Window step size (default = 5)
    * direction STR
        Direction of rolling window: forward = start to end /low to high indices, reverse = end to start/ high to low indices
    """

    n_signal = len(signal)
    n_windows_signal = ((n_signal-win_size)//win_step)+1
    windows = np.empty(dtype=signal.dtype, shape=n_windows_signal)
    window_indices = []

    # Iterate window by window over the signal array and compute the median/mean for each
    for i, j in enumerate(np.arange(0, n_signal-win_size+1, win_step)):

        # evaluate string in smooth as numpy function (median or mean)
        if direction == "forward":
            index = j
            indices = list(range(index, index + win_size))
            window_indices.append(indices)
            # windows[i] = getattr(np,func)(signal[index : index + win_size])
            windows[i] = MAD(signal[index: index + win_size])

        elif direction == "reverse":
            index = n_signal - j
            indices = list(range(index - win_size, index))[::-1]
            window_indices.append(indices)
            # windows[i] = getattr(np,func)(signal[index -win_size : index][::-1])
            windows[i] = MAD(signal[index - win_size: index][::-1])

    return(windows, window_indices)

# ## Function fourierHR(interpolated_signal, time_domain, heart_range = (0.5, 5))
# Perform a Fourier Transform on interpolated signal from heart region


def fourierHR(interpolated_signal, time_domain, heart_range=(0.5, 5)):
    """
    When 3 or less peaks detected is Fourier,
    the true heart-rate is usually the first one.
    The second peak may be higher in some circumstances
    if BOTH chambers were segmented.
    Fourier seems to detect frequencies corresponding to
    1 beat, 2 beats and/or 3 beats in this situation.
    """

    # Fast fourier transform
    fourier = np.fft.fft(interpolated_signal(time_domain))
    # Power Spectral Density
    psd = np.abs(fourier) ** 2

    N = interpolated_signal(time_domain).size
    timestep = np.mean(np.diff(time_domain))
    freqs = np.fft.fftfreq(N, d=timestep)

    # one-sided Fourier spectra
    psd = psd[freqs > 0]
    freqs = freqs[freqs > 0]

    # Calculate ylims for xrange 0.5 to 5 Hz
    heart_indices = np.where(np.logical_and(
        freqs >= heart_range[0], freqs <= heart_range[1]))[0]

    # Peak Calling on Fourier Transform data
    peaks, _ = find_peaks(psd, prominence=1, distance=5)

    # Filter out peaks lower than 1
    peaks = peaks[psd[peaks] >= 5000]  # for detrended and interpolated

    n_peaks = len(peaks)
    if n_peaks > 0:
        # Determine the peak within the heart range
        max_peak = max(psd[peaks])

        # Filter peaks based on ratio to largest peak
        filtered_peaks = peaks[psd[peaks] >= (max_peak * 0.25)]

        # Calculate heart rate in beats per minute (bpm) from the results of the Fourier Analysis
        n_filtered = len(filtered_peaks)
        if n_filtered > 0:

            # Find overlap and sort
            beat_indices = list(set(filtered_peaks) & set(heart_indices))
            beat_indices.sort()

            beat_psd = psd[beat_indices]
            beat_freqs = freqs[beat_indices]

            # If only one peak
            if len(beat_indices) == 1:

                beat_freq = beat_freqs[0]
                beat_power = beat_psd[0]

                bpm = beat_freq * 60
                peak_coord = (beat_freq, beat_power)

            # TODO
            # Need something more sophisticated here
            # If 4 peaks or less, take the first one
            elif 1 < len(beat_indices) < 4:
                beat_freq = beat_freqs[0]
                beat_power = beat_psd[0]

                bpm = beat_freq * 60
                peak_coord = (beat_freq, beat_power)

    # Create dummy bpm variable if doesn't exist
    if "bpm" not in locals():
        bpm = None
        peak_coord = None

    return(psd, freqs, peak_coord, bpm)

# ## Function plotFourier(psd, freqs, peak, bpm, heart_range, figure_loc = 211)
# Plot Fourier Transform


def plotFourier(psd, freqs, peak, bpm, heart_range, figure_loc=211):

    # Prepare label for plot
    if bpm is not None:
        bpm_label = "Heart rate = " + str(int(bpm)) + " bpm"
    else:
        bpm_label = "Heart rate = NA"

    ax = plt.subplot(figure_loc)

    # Plot frequency of Fourier Power Spectra
    _ = ax.plot(freqs, psd)

    # Plot frequency peak if given
    if peak is not None:
        # Put x on freq that correpsonds to heart rate
        _ = ax.plot(peak[0], peak[1], "x")
        # Dotted line to peak
        _ = ax.vlines(x=peak[0], ymin=0, ymax=peak[1], linestyles="dashed")
        _ = ax.hlines(y=peak[1], xmin=0, xmax=peak[0], linestyles="dashed")
        # Annotate with BPM
        _ = ax.annotate(bpm_label, xy=peak, xytext=(
            peak[0] + 0.5, peak[1] + (peak[1] * 0.05)), arrowprops=dict(facecolor='black', shrink=0.05))

    # Only plot within heart range (in Hertz) if necessary
    if heart_range is not None:
        _ = ax.set_xlim(heart_range)

    _ = ax.set_ylim(top=max(psd) + (max(psd) * 0.2))

    # Y-axis label
    _ = ax.set_ylabel('Power Spectra')

    return(ax)

# ## Function def PixelSignal(grey_frames)


def PixelSignal(grey_frames):
    """
        Extract individual pixel signal across all frames
    """
  # pixel_num = 0
    # pixel_signals = {}

    # #First non-empty frame
    # first = next(x for x, frame in enumerate(grey_frames) if frame is not None)
    # rows, cols = grey_frames[first].shape
    # for i in range(rows):
    #     for j in range(cols):
    #         pixel_vals = []
    #         for frame in grey_frames:
    #             if frame is not None:
    #                 pixel = frame[i,j]
    #                 pixel_vals.append(pixel)

    #             else:
    #                 pixel_vals.append(np.nan)

    #         #Filter out masked pixels,
    #         #i.e. those with a sum of zero
    #         pixel_sum = np.nansum(pixel_vals)
    #         if pixel_sum > 0:
    #             pixel_signals[pixel_num] = pixel_vals

    #         pixel_num += 1

    # #TODO Remove dictionary backwards compatibility, as it's just unneccassarily slow
    as_array = np.asarray(grey_frames)
    pixel_signals = np.transpose(
        as_array, axes=[1, 2, 0]).reshape(-1, as_array.shape[0])
    pixel_signals = dict(enumerate(pixel_signals.tolist()))

    remove = [k for k in pixel_signals if all(
        v == 0 for v in pixel_signals[k])]
    for k in remove:
        del pixel_signals[k]

    return(pixel_signals)

# Plot multiple pixel signal intensities on same graph


def PixelFourier(pixel_signals, times, empty_frames, frame2frame, threads, pixel_num=None, heart_range=(0.5, 5), plot=False):
    """
    Plot multiple pixel signal intensities on same graph
    * pixel_signals DICT
        Dictionary of pixel signal intensities
    """

    increment = frame2frame / 6
    td = np.arange(start=times[0], stop=times[-1] + increment, step=increment)

    # Randomly select N pixels in to determine heart-rate from
    # Limit to N pixels in interest of speed
    if pixel_num is not None:
        if len(pixel_signals) >= pixel_num:

            # Set seed
            random.seed(42)

            # Select 1000 (pseudo-)random pixels
            selected_pixels = random.sample(list(pixel_signals.keys()), k=1000)
        else:
            selected_pixels = list(pixel_signals.keys())

    # If number of pixels not specified, calculate for all pixels
    else:
        selected_pixels = list(pixel_signals.keys())

    if plot is True:

        ax = plt.subplot()

        # Only plot within heart range (in Hertz)
        _ = ax.set_xlim(heart_range)

        _ = ax.set_xlabel('Frequency (Hz)')
        _ = ax.set_ylabel('Power Spectra')

        highest_freqs = []
        # Plot signals for selected pixels
        for pixel in selected_pixels:

            pixel_signal = pixel_signals[pixel]

            # Remove values from empty frames
            signal_filtered = np.delete(pixel_signal, empty_frames)
            times_filtered = np.delete(times, empty_frames)

            # Cubic Spline Interpolation
            cs = CubicSpline(times_filtered, signal_filtered)
            norm_cs = detrendSignal(cs, td, window_size=27)   # 27

            # Fourier on interpolated pixel signal
            psd, freqs, _, _ = fourierHR(norm_cs, td)

            # Determine the peak within the heart range
            heart_indices = np.where(np.logical_and(
                freqs >= heart_range[0], freqs <= heart_range[1]))[0]

            # Spectra within heart range
            heart_psd = psd[heart_indices]
            heart_freqs = freqs[heart_indices]

            # Index of largest spectrum in heart range
            index_max = np.argmax(heart_psd)
            # Corresponding frequency
            highest_freq = heart_freqs[index_max]
            highest_freqs.append(highest_freq)

            # Plot frequency of Fourier Power Spectra
            _ = ax.plot(freqs, psd, color='grey', alpha=0.5)

        return(ax, highest_freqs)

    # Run all pixels without plotting them individually
    else:
        highest_freqs = Parallel(n_jobs=threads, prefer="threads")(delayed(PixelNorm)(
            pixel, pixel_signals, times, empty_frames, heart_range, td) for pixel in selected_pixels)

        return(highest_freqs)

# ## Function PixelNorm(pixel, pixel_signals, times, empty_frames, heart_range, plot = False):


def PixelNorm(pixel, pixel_signals, times, empty_frames, heart_range, td, plot=False):

    pixel_signal = pixel_signals[pixel]

    # Remove values from empty frames
    signal_filtered = np.delete(pixel_signal, empty_frames)
    times_filtered = np.delete(times, empty_frames)

    # Cubic Spline Interpolation
    cs = CubicSpline(times_filtered, signal_filtered)
    norm_cs = detrendSignal(cs, td, window_size=27)  # 27

    # Fourier on interpolated pixel signal
    psd, freqs, _, _ = fourierHR(norm_cs, td)

    # Determine the peak within the heart range
    heart_indices = np.where(np.logical_and(
        freqs >= heart_range[0], freqs <= heart_range[1]))[0]

    # Spectra within heart range
    heart_psd = psd[heart_indices]
    heart_freqs = freqs[heart_indices]

    # Index of largest spectrum in heart range
    index_max = np.argmax(heart_psd)
    # Corresponding frequency
    highest_freq = heart_freqs[index_max]

#   cs_psd = CubicSpline(freqs, psd)

    # Plot pixel PSD
#   if plot is True:
    # Plot pixel PSD
#       _ = ax.plot(freqs, psd, color='grey', alpha=0.5)

    return(highest_freq)

# ## Function PixelFreqs(frequencies, figsize = (10,7), heart_range = (0.5, 5), peak_filter = True)


def PixelFreqs(frequencies, average_values, figsize=(10, 7), heart_range=(0.5, 5), peak_filter=True):

    sns.set_style('white')
    ax = plt.subplot()

    peak_variance = np.var(frequencies)
    # Deal with very homogeneous array of Fourier Peaks,
    # otherwise variance will be too small for KDE.
    # Needs to be greater than 0
    if peak_variance < 0.0001:

        LOGGER.info("In PixelFrequency analysis: low variance")

        # Take mode of (homogeneous) array to be the heart rate
        median_peaks = np.median(frequencies)
        bpm = median_peaks * 60
        bpm = np.around(bpm, decimals=2)

        # Prepare label for plot
        bpm_label = str(int(bpm)) + " bpm"

        hist_height = max([h.get_height() for h in sns.histplot(
            frequencies, stat='density', bins=500).patches])

        # Plot Histogram
        _ = sns.histplot(frequencies, ax=ax, fill=True,
                         stat='density', bins=500)

        _ = ax.set_title("Pixel Fourier Transform Maxima")
        _ = ax.set_xlabel('Frequency (Hz)')
        _ = ax.set_ylabel('Density')
        # Only plot within heart range (in Hertz)
        _ = ax.set_xlim(heart_range)

        _ = ax.annotate(bpm_label, xy=(median_peaks, hist_height), xytext=(median_peaks + (median_peaks * 0.1),
                        hist_height + (hist_height * 0.01)), arrowprops=dict(facecolor='black', shrink=0.05))

    else:
        # Detect most common Fourier Peak using Kernel Density Estimation (KDE)
        density = gaussian_kde(frequencies)
        xs = np.linspace(heart_range[0], heart_range[-1], 500)
        ys = density(xs)

        # Plot KDE
        _ = sns.kdeplot(frequencies, ax=ax, fill=True)  # , bw_adjust=.5)
        _ = ax.set_title("Pixel Fourier Transform Maxima")
        _ = ax.set_xlabel('Frequency (Hz)')
        _ = ax.set_ylabel('Density')
        # Only plot within heart range (in Hertz)
        _ = ax.set_xlim(heart_range)

        # Calculate bpm from most common Fourier peak
        max_index = np.argmax(ys)
        # min_index = np.argmin(ys)
        # sorted_list = ys.sort()
        # max_y = sorted_list[-1]

        # let´s find the max and min peaks, and use it in case we have two peaks and user has input an average as argument to filter peaks

        max_x = xs[max_index]
        max_y = ys[max_index]

        # Peak Calling
        # prominence filters out 'flat' KDEs,
        # these result from a noisy signal
        if peak_filter is True:
            peaks, _ = find_peaks(ys, prominence=0.5)
        else:
            peaks, _ = find_peaks(ys, prominence=0.1)

        if len(peaks) == 1:
            LOGGER.info("Found 1 peak")
            bpm = max_x * 60
            bpm = np.around(bpm, decimals=2)

            # Prepare label for plot
            bpm_label = str(int(bpm)) + " bpm"

            # Label plot with bpm
            _ = ax.plot(max_x, max_y, 'bo', ms=10)
            _ = ax.annotate(bpm_label, xy=(max_x, max_y), xytext=(
                max_x + (max_x * 0.1), max_y + (max_y * 0.01)), arrowprops=dict(facecolor='black', shrink=0.05))

        elif len(peaks) > 1:
            # verify if user has inserted a average argument -a. 0 means No parameters inserted
            if not average_values:
                LOGGER.info("found " + str(len(peaks)) +
                            " peak(s), selected the highest one")
                LOGGER.info(
                    "in these cases, inserting an expected average as agument -a in bash command line can help to choose the right peak. E.g.: -a 98")
                bpm = max_x * 60
                bpm = np.around(bpm, decimals=2)

                bpm_label = str(int(bpm)) + " bpm"
                # Label plot with bpm
                _ = ax.plot(max_x, max_y, 'bo', ms=10)
                _ = ax.annotate(bpm_label, xy=(max_x, max_y), xytext=(
                    max_x + (max_x * 0.1), max_y + (max_y * 0.01)), arrowprops=dict(facecolor='black', shrink=0.05))

            else:
                # the user insert, as an parameter -a value (average), then the algorithm detect peaks and consider the peak closest to the average

                # list_from_array = xs.tolist()
                # list_from_array_y_index = [x for x in range(len(ys)) if ys[x] in list_from_array_y]

                x_values = [xs[i] for i in peaks]

                def absolute_difference_function(list_value): return abs(
                    list_value - average_values/60)
                closest_value = min(x_values, key=absolute_difference_function)

                x_list = xs.tolist()

                index_for_plotting = x_list.index(closest_value)

                bpm = closest_value * 60
                bpm = np.around(bpm, decimals=2)

                bpm_label = str(int(bpm)) + " bpm"
                # Label plot with bpm
                _ = ax.plot(xs[index_for_plotting],
                            ys[index_for_plotting], 'bo', ms=10)
                _ = ax.annotate(bpm_label, xy=(xs[index_for_plotting], ys[index_for_plotting]), xytext=(xs[index_for_plotting] + (
                    xs[index_for_plotting] * 0.1), ys[index_for_plotting] + (ys[index_for_plotting] * 0.01)), arrowprops=dict(facecolor='black', shrink=0.05))

                LOGGER.info("found " + str(len(peaks)) + " peak(s), and selected the closest to " +
                            str(average_values) + ', as requested in the argument -a')

        else:
            bpm = None
            LOGGER.info('No peaks detected, as prominence is < 0.1')
            # print('ERROR code 3')

    return(ax, bpm)

# ## Function output_report(well_number, well, bpm_fourier = "NA", error_message = "No message"):

# #create an final report or append tbpm result to an existent one
# def output_report(well_number, well, bpm_fourier = "NA", error_message = "No message"):
#     output_dir = out_base + "/general_report.csv"
#     #print(output_dir)
#     try:
#         f = open(output_dir)
#         f.close()

#         with open(output_dir, 'a') as results:
#             final_file = csv.writer(results, delimiter=',', quotechar='"', lineterminator = '\n')
#             final_file.writerow([well_number, well, bpm_fourier, error_message])

#     except IOError:
#         with open(output_dir, 'a') as results:
#             final_file = csv.writer(results, delimiter=',', quotechar='"', lineterminator = '\n')
#             final_file.writerow(['well', 'twell_id', 'tbpm', 'message if any'])
#             final_file.writerow([well_number, well, bpm_fourier, error_message])

#     return

# ## Function final_dist_graph(bpm_fourier)
# Create an graph and plot averages from final report (overwrite and update if alreadye exists)


def final_dist_graph(bpm_fourier,  out_dir):
    data_file = os.path.join(out_dir, "general_report.csv")
    output_dir = os.path.join(out_dir, "data_dist_plot.jpg")

    try:
        data = pd.read_csv(data_file, usecols=[2])
        # data = data.dropna()
        # data = pd.read_csv(resulting_reports + loop + '/' + frame_format + "_general_report.csv"", usecols=[2])

        # calculate total rows and valid rows so we can calculate error index an others indicators
        number_total_of_rows = len(data)
        # print(number_total_of_rows)
        data = data.dropna()
        valid_number_of_rows = len(data)

        rows_with_error = number_total_of_rows - valid_number_of_rows

        # calculate error index
        error_index = rows_with_error/number_total_of_rows*100
        # print(data)

        # calculate Coeficient of variation
        list = data['tbpm'].to_list()
        def cv(x): return np.std(x, ddof=1) / np.mean(x) * 100
        cv = round(cv(list), 2)
        mean_data = round(statistics.mean(list), 2)

        std_dev = round(np.std(list), 2)
        # log10 = np.log10(list)
        # log10_round_to = [round(num, 2) for num in log10]

        final_text = ("total_rows: " + str(number_total_of_rows) + ";  "
                      + "Error rows: " + str(rows_with_error) + ";  "
                      + "error index: " + str(round(error_index, 2))
                      + " %;  " + "average: " + str(mean_data)
                      + ";  std deviation: " + str(std_dev)
                      + ";  variation coeficient: " + str(cv))
        sns.set(font_scale=1.2)
        fig, axs = plt.subplots(ncols=2, figsize=(15, 10))

        double_plot = sns.histplot(data=list, bins=10, ax=axs[0])  # data.tbpm
        ymin, ymax = double_plot.get_ylim()
        double_plot.set(ylim=(ymin, ymax+24))
        for p in double_plot.patches:
            double_plot.annotate(format(p.get_height(), 'd'), (p.get_x() + p.get_width() / 2.,
                                 p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
        double_plot.axes.set_title(
            "Frequency Histogram of tbpm data", fontsize=16)
        double_plot = sns.boxplot(
            x=data["tbpm"], orient="h", color='salmon', ax=axs[1])
        double_plot.axes.set_title("Boxplot of tbpm data", fontsize=16)
        fig.suptitle(final_text, fontsize=12)
        fig = double_plot.get_figure()
        fig.savefig(output_dir)
        plt.close(fig)

    except:
        pass

# final_dist_graph(bpm_fourier)    ## debug


def embryo_detection(video):
    center_of_embryo_list = []
    for img, i in zip(video, range(5)):

        # norming
        max_of_img = np.max(img)

        img = np.uint8(img / max_of_img * 255)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blurred_img = cv2.GaussianBlur(gray_img, (501, 501), 200.0)

        # Divide: i2 = (i1/i2) x k1 + k2] k1=8000 k2=0" NOTE: 120 worked here, not sure why imagej script uses 8000. Might be because of following z-projection, which is applied to the image stack.
        img = np.divide(gray_img, blurred_img) * 120

        # thresholding
        thresh_img = cv2.GaussianBlur(img, (25, 25), 10)
        thresh = threshold_yen(thresh_img)
        thresh_img = thresh_img > thresh
        thresh_img_final = thresh_img*255

        # clear 10% of the image' borders as some dark areas may exists
        thresh_img_final[0:int(thresh_img_final.shape[1]*0.1),
                         0:thresh_img_final.shape[0]] = 255
        thresh_img_final[int(thresh_img_final.shape[1]*0.9):thresh_img_final.shape[1], 0:thresh_img_final.shape[0]] = 255

        thresh_img_final[0:thresh_img_final.shape[1],
                         0:int(thresh_img_final.shape[0]*0.1)] = 255
        thresh_img_final[0:thresh_img_final.shape[1], int(
            thresh_img_final.shape[0]*0.9):thresh_img_final.shape[0]] = 255

        # Transform 0/255 image to 0/1 image
        thresh_img_final[thresh_img_final > 0] = 1

        # invert image
        image_inverted = np.logical_not(thresh_img_final).astype(int)

        # calculate the center of mass of inverted image
        count = (image_inverted == 1).sum()
        x_center, y_center = np.argwhere(
            image_inverted == 1).sum(0)/count

        center_of_embryo_list.append((x_center, y_center))

    XY_average = (mean([i[0] for i in center_of_embryo_list]), mean(
        [i[1] for i in center_of_embryo_list]))

    return XY_average


def crop_2(video, args, embryo_coordinates, resulting_dict_from_crop, video_metadata):
    # avoid window size lower than 50 or higher than the minimum dimension of images
    # window size is the size of the window that the script will crop starting from centre os mass,
    # and can be passed as argument in command line (100 is default)
    embryo_size = args.embryo_size
    # get the minimum size of the first frame
    maximum_dimension = min(video[0].shape[0:1])
    if embryo_size < 50:
        embryo_size = 50
    if embryo_size > int((maximum_dimension/3)):
        embryo_size = int((maximum_dimension/3))
        LOGGER.info(
            "-s paramter has excedded the allowed by image dimensions. Used " + str(embryo_size) + " instead.")
    #embryo_size += 100

    video_cropped = []

    for index, img in enumerate(video):
        try:
            cut_image = img[int(embryo_coordinates[0])-embryo_size: int(embryo_coordinates[0]) +
                            embryo_size, int(embryo_coordinates[1])-embryo_size: int(embryo_coordinates[1])+embryo_size]
        except:
            cut_image = img
            LOGGER.info(
                "Problems cropping image (image dimensions in -s paramter)")

        video_cropped.append(cut_image)

        # create a dictionary with all first image from every well. This dictionary will be persistent across the functions calls
        if index == 0:
            if video_metadata['channel'] + '_' + video_metadata['loop'] not in resulting_dict_from_crop:
                resulting_dict_from_crop[video_metadata['channel'] +
                                         '_' + video_metadata['loop']] = [cut_image]
                resulting_dict_from_crop['positions_' + video_metadata['channel'] +
                                         '_' + video_metadata['loop']] = [video_metadata['well_id']]
            else:
                resulting_dict_from_crop[video_metadata['channel'] +
                                         '_' + video_metadata['loop']].append(cut_image)
                resulting_dict_from_crop['positions_' + video_metadata['channel'] +
                                         '_' + video_metadata['loop']].append(video_metadata['well_id'])

       # return the cropped video array and the dictionary with data from every first cropped video updated
    return video_cropped, resulting_dict_from_crop
############################################################################################################

# DEPRECATED: substituted by crop_2 (above)


def crop(video):
    LOGGER.info("Cropping video")
    circle_x = []
    circle_y = []
    circle_radii = []

    video_cropped = []

    success_count = 0
    for frame, i in zip(video, range(5)):
        # Detect embryo with Hough Circle Detection
        # Circle coords, can be used as cropping parameters
        circle, x1, y1, x2, y2 = detectEmbryo(frame)

        if circle is not None:
            circle_x.append(circle[0])
            circle_y.append(circle[1])
            circle_radii.append(circle[2])

    # Embryo Circle
    if len(circle_x) > 0:
        for x, y, r in zip(circle_x, circle_y, circle_radii):

            x_coord = x
            y_coord = y
            radius = r

            if x_coord is not None:

                x_counts = Counter(x_coord)
                y_counts = Counter(y_coord)
                rad_counts = Counter(radius)

        # Use most common circle coords for the embryo
        embryo_x, _ = x_counts.most_common(1)[0]
        embryo_y, _ = y_counts.most_common(1)[0]
        embryo_rad, _ = rad_counts.most_common(1)[0]

    else:
        embryo_x = None
        embryo_y = None
        embryo_rad = None

    if embryo_x and embryo_y and embryo_rad:
        y1 = embryo_y - embryo_rad - 50
        y2 = embryo_y + embryo_rad + 50
        x1 = embryo_x - embryo_rad - 50
        x2 = embryo_x + embryo_rad + 50

    else:
        LOGGER.warning(
            "Couldn't detect circles in video. Cutting approximately around edges.")
        shape = video[0].shape

        y1 = int(shape[0] * 0.3)
        y2 = int(shape[0] * 0.7)
        x1 = int(shape[1] * 0.3)
        x2 = int(shape[1] * 0.7)
    for frame in video:
        # crop_img = img[y1 : y2, x1: x2]
        video_cropped.append(frame[y1: y2, x1: x2])

    return video_cropped


# TODO: Too long, should probably be split into separate parts to maintain overview of steps in the algorithm pipeline
# run the algorithm on a well video
def run(video, args, video_metadata):
    LOGGER.info("Starting algorithmic analysis")
    # Read all images for a given well
    # >0 returns a 3 channel RGB image. This results in conversion to 8 bit.
    # 0 returns a greyscale image. Also results in conversion to 8 bit.
    # <0 returns the image as is. This will return a 16 bit image.

    # store the out_dir without the well positiom, so we can use it to store general reports in main folder
    out_base = args['outdir']

    # Add well position to output directory path
    out_dir = os.path.join(args['outdir'], video_metadata['channel'],
                           video_metadata['loop'] + '-' + video_metadata['well_id'])

    os.makedirs(out_dir, exist_ok=True)

    sorted_frames = video
    frame_dict = {}
    sorted_times = video_metadata['timestamps']

    for img, time in zip(sorted_frames, sorted_times):

        if img is not None:
            frame_dict[time] = img
        else:
            frame_dict[time] = None

    # Determine frame rate from time-stamps if unspecified
    if not args['fps']:
        # total time acquiring frames in seconds
        timestamp0 = int(sorted_times[0])
        timestamp_final = int(sorted_times[-1])
        total_time = (timestamp_final - timestamp0) / 1000
        # fps = int(len(sorted_times) / round(total_time))
        fps = len(sorted_times) / total_time
        LOGGER.info("fps detected: " + str(fps))
    else:
        fps = args['fps']
        LOGGER.info("fps informed: " + str(fps))

    # Remove duplicate time stamps,
    # same frame can have been saved more than once
    sorted_times = list(OrderedDict.fromkeys(sorted_times))
    sorted_frames = [frame_dict[time] for time in sorted_times]

    LOGGER.info("Normalizing frames")
    # Normalize frames
    norm_frames = normVideo(sorted_frames)

    LOGGER.info("Writing video")

    # Write video
    vid_frames = [frame for frame in norm_frames if frame is not None]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'avc1')

    try:
        height, width, layers = vid_frames[0].shape
    except IndexError:
        height, width = vid_frames[0].shape

    size = (width, height)
    out_vid = os.path.join(out_dir, "embryo.mp4")

    out = cv2.VideoWriter(out_vid, fourcc, fps, size)

    for i in range(len(vid_frames)):
        out.write(vid_frames[i])
    out.release()

    LOGGER.info("Detecting HROI")
    # Only process if less than 5% frames are empty
    if sum(frame is None for frame in sorted_frames) > len(sorted_frames) * 0.05:
        raise ValueError("More than 5% of frames are empty")

    embryo = []

    # Start from first non-empty frame
    start_frame = next(x for x, frame in enumerate(
        norm_frames) if frame is not None)

    # new feature to filter mask
    hsvz = cv2.cvtColor(norm_frames[start_frame], cv2.COLOR_RGB2HSV)
    lower_green = np.array([0, 0, 0])
    upper_green = np.array([150, 150, 150])
    maskx = cv2.inRange(hsvz, lower_green, upper_green)
    kernelz = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    opened_maskz = cv2.morphologyEx(maskx, cv2.MORPH_OPEN, kernelz)
    plt.imshow(norm_frames[start_frame])
    plt.imshow(opened_maskz)

    # Add None if first few frames are empty
    empty_frames = range(start_frame)
    for i in empty_frames:
        embryo.append(None)

    frame0 = norm_frames[start_frame]
    embryo.append(frame0)

    # Generate blank images for masking
    rows, cols, _ = frame0.shape
    heart_roi = np.zeros(shape=[rows, cols], dtype=np.uint8)
    blue_print = np.zeros(shape=[rows, cols], dtype=np.uint8)

    # Process frame0
    _, old_grey, _ = processFrame(frame0)
    f0_grey = old_grey.copy()

    # Numpy matrix:
    # Coord 1 = row(s)
    # Coord 2 = col(s)

    # Detect heart region (and possibly blood vessels)
    # by determining the differences across windows of frames
    j = start_frame + 1
    while j < len(norm_frames):

        frame = norm_frames[j]

        if frame is not None:

            masked_frame, triangle_thresh, thresh, movement = rolling_diff(
                j, norm_frames, win_size=2, direction="reverse", min_area=150)  # I´ve added thresh just to test

            # TODO: If the movement happens early, all data is thrown away and the algorithm fails.
            # if movement is true, it means that the embyio flip around. Then, save the frame number and and skip all following frames. The frame number will be used to try run the fast method if is the case
            if movement == True:
                stop_frame = j
                break

            heart_roi = cv2.add(heart_roi, triangle_thresh)
            embryo.append(masked_frame)
            cv2.add(thresh, blue_print)

            # cv2.imwrite('color_img.jpg', heart_roi)

        else:
            embryo.append(None)
        j += 1

    heart_roi = cv2.bitwise_and(heart_roi, heart_roi, mask=opened_maskz)

    # Get indices of 250 most changeable pixels
    top_pixels = 250
    changeable_pixels = np.unravel_index(np.argsort(
        heart_roi.ravel())[-top_pixels:], heart_roi.shape)
    # changeable_pixels = np.unravel_index(np.argsort(thresh.ravel())[-top_pixels:], thresh.shape)

    # Create boolean matrix the same size as the RoI image
    maxima = np.zeros((heart_roi.shape), dtype=bool)

    # Label pixels based on based on the top changeable pixels
    maxima[changeable_pixels] = True
    label_maxima = label(maxima)

    # Perform 'opening' on heat map of absolute differences
    # Rounds of erosion and dilation
    heart_roi_open = cv2.morphologyEx(heart_roi, cv2.MORPH_OPEN, kernel)

    # Threshold heart RoI to generate mask
    yen = threshold_yen(heart_roi_open)
    heart_roi_clean = heart_roi_open > yen
    heart_roi_clean = heart_roi_clean.astype(np.uint8)

    # Filter mask based on area of contours
    heart_roi_clean = filterMask(mask=heart_roi_clean, min_area=100)

    final_mask, regions = new_qc_mask_contours(heart_roi_clean, maxima)
    # final_mask, regions = qc_mask_contours(heart_roi_clean, maxima, top_pixels)

    # output_report(well_number, well, error_message = "no masks")
    # print(nothing_here_just_to_throw_an_error)

    overlay = color.label2rgb(label_maxima, image=final_mask,
                              alpha=0.7, bg_label=0, bg_color=None, colors=[(1, 0, 0)])

    # Generate figure showing with potential heart region highlighted
    out_fig = os.path.join(out_dir, "embryo_heart_roi.png")
    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    ax = heartQC_plot(ax, f0_grey, heart_roi, heart_roi_clean, overlay)
    plt.savefig(out_fig, bbox_inches='tight')
    plt.show()
    plt.close()

    # Check if heart region was detected, i.e. if sum(masked) > 0
    # and limit number of possible heart regions to 3 or fewer
    # if (final_mask.sum() > 0) and (regions <= 3):
    if (final_mask.sum() <= 0):
        raise RuntimeError("Couldn't detect a HROI")

    mask = final_mask

    # Coefficient of variation
    cvs = {}
    times = []

    timestamp0 = int(sorted_times[0])

    LOGGER.info("Statistical analysis")
    masked_frames = []
    # Draw contours of heart
    for i in range(len(embryo)):

        # raw_frame = sorted_frames[i]
        frame = embryo[i]

        if frame is not None:
            masked_data = cv2.bitwise_and(frame, frame, mask=mask)
            # TODO: Suspected source of errors. Check and inspect this for frames after the first.
            masked_grey = cv2.cvtColor(masked_data, cv2.COLOR_BGR2GRAY)

            # print('masked_grey')
            # print('Embryo number: ' + str(i))

            # Create vector of the signal within the region of the heart
            # Flatten array (matrix) into a vector
            heart_values = np.ndarray.flatten(masked_grey)
            # Remove zero elements
            heart_values = heart_values[np.nonzero(heart_values)]

            # Mean signal in region
            heart_mean = np.mean(heart_values)
            # Standard deviation for signal in region
            heart_std = np.std(heart_values)
            # Coefficient of variation
            heart_cv = heart_std / heart_mean

            # split source frame into B,G,R channels
            b, g, r = cv2.split(frame)

            # add a constant to B (blue) channel to highlight the heart
            b = cv2.add(b, 100, dst=b, mask=mask, dtype=cv2.CV_8U)

            masked_frame = cv2.merge((b, g, r))

            #################################

        # No signal in heart RoI if the frame is empty
        else:
            masked_frame = None
            masked_grey = None
            heart_std = np.nan
            heart_cv = np.nan

        masked_frames.append(masked_grey)
        embryo[i] = masked_frame

        frame_num = i + 1
        cvs[frame_num] = heart_cv

        # Calculate time between frames
        # Time frame 1 = 0 secs
        if i == 0:
            time = 0
            time_elapsed = 0

            if masked_frame is not None:
                # Save first frame with the ROI highlighted
                out_fig = os.path.join(out_dir, "masked_frame.png")
                cv2.imwrite(out_fig, masked_frame)

        # Time frame i = (frame i - frame i-1) / 1000
        # Total Time frame i = (frame i - frame 0) / 1000
        else:
            timestamp = int(sorted_times[i])
            old_timestamp = int(sorted_times[i-1])
            # Time between frames in seconds
            time = (timestamp - old_timestamp) / 1000
            # Total time elapsed in seconds
            time_elapsed = (timestamp - timestamp0) / 1000

        times.append(time_elapsed)

    # Write video
    out_vid = os.path.join(out_dir, "embryo_changes.mp4")
    vid_frames = [i for i in embryo if i is not None]
    height, width, layers = vid_frames[0].shape
    size = (width, height)
    out2 = cv2.VideoWriter(out_vid, fourcc, fps, size)
    for i in range(len(vid_frames)):
        out2.write(vid_frames[i])
    out2.release()

    ############################
    # Quality control heart rate estimate
    ############################
    # Min and max bpm from Jakob paper
    y = np.asarray(list(cvs.values()), dtype=float)

    # Get indices of na values
    na_values = np.isnan(y)
    empty_frames = [i for i, x in enumerate(na_values) if x]

    ##############################################
    # if fps is given, calculate differently
    if args['fps']:
        frame2frame = 1 / fps
        final_time = frame2frame * len(times)
        times = np.arange(start=0, stop=final_time, step=frame2frame)
    else:
        times = np.asarray(times, dtype=float)
        frame2frame = np.mean(np.diff(times))

    # Time domain for interpolation
    increment = frame2frame / 6
    td = np.arange(start=times[0], stop=times[-1] + increment, step=increment)

    # Filter, detrend and interpolate region signal
    # print('#Filter, detrend and interpolate region signal')
    # print(times)
    # print(y)
    # print(empty_frames)

    # TODO: This is unused atm, and fails
    times_final, y_final, cs = interpolate_signal(times, y, empty_frames)
    meanY = np.mean(cs(td))

    LOGGER.info("Fourier frequency evaluation")

    # Run normally, Fourier in segemented area
    if not args['slowmode']:

        # Signal per pixel
        pixel_signals = PixelSignal(masked_frames)

        # print('not slow')
        # Perform Fourier Transform on each pixel in segmented area
        out_fourier = os.path.join(out_dir, "pixel_fourier.png")
        fig, ax = plt.subplots(figsize=(10, 7))
        ax, highest_freqs = PixelFourier(
            pixel_signals, times, empty_frames, frame2frame, args['threads'], pixel_num=1000, plot=True)
        plt.savefig(out_fourier)

        # plt.imshow(ax)
        plt.show()

        plt.close()

        # Plot the density of fourier transform global maxima across pixels
        out_kde = os.path.join(out_dir, "pixel_rate.png")
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        ax, bpm_fourier = PixelFreqs(
            highest_freqs, args['average'], peak_filter=True)
        plt.savefig(out_kde)
        plt.close()

    # Run in slow mode, Fourier on every pixel
    # must be removed from here and sent to the end, as if the script does not find any mask, it still runs the slow method
    else:
        LOGGER.info("Running in slow mode")
        # All pixels
        norm_frames_grey = greyFrames(norm_frames)
        # Resize frames to make faster
        norm_frames_grey = resizeFrames(norm_frames_grey, scale=50)
        # Signal for every pixel
        all_pixel_sigs = PixelSignal(norm_frames_grey)

        # Perform Fourier Transform on every pixel
        highest_freqs2 = PixelFourier(
            all_pixel_sigs, times, empty_frames, frame2frame, args['threads'], plot=True)

        # print(highest_freqs2)
        # print(erro)

        # Plot the density of fourier transform global maxima across pixels
        out_kde2 = os.path.join(out_dir, "pixel_rate_all.png")
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 7))
        ax2, bpm_fourier = PixelFreqs(
            highest_freqs2, args['average'], peak_filter=False)
        plt.savefig(out_kde2)
        plt.close()

    # Calculate slope
    # Presumably should be flat(ish) if good
    # Or fit line
#       slope, intercept, r_value, p_value, std_err = stats.linregress(td, cs(td))

    # Median Absolute Deviation across signal
#       mad = stats.median_absolute_deviation(cs(td))

    # Frequently issue with first few data-points
    to_keep = range(int(len(td) * 0.05), len(td))
    filtered_td = td[to_keep]

    # Plot sigal in segmented region
    out_fig = os.path.join(out_dir, "bpm_trace.png")
    plt.figure(figsize=(10, 2))
    plt.plot(filtered_td, cs(filtered_td))
    plt.ylabel('Heart intensity (CoV)')
    plt.xlabel('Time [sec]')
    plt.hlines(y=np.mean(cs(filtered_td)),
               xmin=td[0], xmax=td[-1], linestyles="dashed")
    plt.savefig(out_fig, bbox_inches='tight')

    plt.show()

    plt.close()

    # Filter out if linear regression captures signal trend well
    # (i.e. if p-value highly significant)
#       if (np.float64(p_value) > np.float_power(10, -8)) or (mad <= 0.02) or (np.absolute(slope) <= 0.002):

    # Detrend and smoothe cubic spline interpolated data
    # with Savitzky–Golay filter
#           norm_cs = detrendSignal(cs, td, window_size = 21)

#           out_fig = out_dir + "/bpm_trace.savgol_filter.png"
#           plt.figure(figsize=(10,2))
#           plt.plot(td, norm_cs(td))
#           plt.ylabel('Normalised Heart Signal')
#           plt.xlabel('Time [sec]')
#           plt.hlines(y = np.mean(norm_cs(td)), xmin = td[0], xmax = td[-1], linestyles = "dashed")
#           plt.savefig(out_fig,bbox_inches='tight')
#           plt.close()

    # Remove first 2.5% of interpolated, detrended data-points
    # Frequently issue with first few data-points
#           to_keep = range(int(len(td) * 0.05),len(td))
#           filtered_td = td[to_keep]

# ## run the report and graphs functions

    # check heartrate was calculated
    # otherwise create variable and set to None
    if "bpm_fourier" not in locals():
        # print('not in locals')
        bpm_fourier = None
        LOGGER.info("Couldn't determine BPM in fast mode")

    # bpm_fourier = "NA"

    # Try to use fast method
    '''if bpm_fourier == 'NA':
        try:
            print('Try to use recovery method')
            bpm_fourier_fast = run_a_well(indir, out_dir, frame_format, well_number, loop)    
            bpm_fourier = round(bpm_fourier_fast.get('bpm'),2)
            print('recoverede:')
            print(bpm_fourier)
            error_message = error_message + ' using the fast method'
        except:
            error_message="ERROR in fast method"'''

    # try to use slow method
    if bpm_fourier is None:
        LOGGER.info("Trying slow mode")
        # All pixels
        if "stop_frame" in locals():
            norm_frames_grey = greyFrames(norm_frames, stop_frame)
        else:
            norm_frames_grey = greyFrames(norm_frames)
        # Resize frames to make faster
        norm_frames_grey = resizeFrames(norm_frames_grey, scale=50)
        # Signal for every pixel
        all_pixel_sigs = PixelSignal(norm_frames_grey)

        highest_freqs2 = PixelFourier(
            all_pixel_sigs, times, empty_frames, frame2frame, args['threads'], plot=False)
        # Plot the density of fourier transform global maxima across pixels
        out_kde2 = os.path.join(out_dir, "pixel_rate_all.png")
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 7))
        ax2, bpm_fourier = PixelFreqs(
            highest_freqs2, args['average'], peak_filter=False)
        plt.savefig(out_kde2)
        plt.close()

    # Create or subscribe the histogram and distgraphs
    # if bpm_fourier is not None:
    #     #print('call graph function')
    #     final_dist_graph(bpm_fourier)

    plt.close('all')
    return bpm_fourier
