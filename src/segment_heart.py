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
import warnings
import math
from matplotlib import pyplot as plt
import statistics
import os
import logging

from statistics import mean

import numpy as np
import cv2

# import skimage
from skimage.filters import threshold_triangle, threshold_yen
from skimage.measure import label
from skimage import color

# import scipy
from scipy.signal import savgol_filter, detrend 
import matplotlib
from mpl_toolkits.mplot3d import axes3d

matplotlib.use('Agg')

# Parallelisation
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=SyntaxWarning)

################################################################################
##########################
##  Globals   ##
##########################

LOGGER = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# Kernel for image smoothing
KERNEL = np.ones((5, 5), np.uint8)

### Cropping Feature
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
        except Exception as e:
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

### Main Algorithm
### Heart Segmentation and frequeny detection
def save_video(video, fps, outdir, filename):
    video = assert_8bit(video)
    
    if(len(video[0].shape) == 2):
        video = np.array([cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) for frame in video])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    try:
        height, width, layers = video[0].shape
    except (IndexError, ValueError):
        height, width = video[0].shape
    size = (width, height)
    out_vid = os.path.join(outdir, filename)
    out = cv2.VideoWriter(out_vid, fourcc, fps, size)

    for i in range(len(video)):
        out.write(video[i])
    out.release()

# ## Function normVideo(frames)
# Normalise across frames to harmonise intensities
# TODO: A single outlier will worsen the normalization. Ignore extreme outliers for the stretching.
def normVideo(frames):
    min_in_frames = np.min(frames)
    max_in_frames = np.max(frames)

    norm_frames = (np.subtract(frames, min_in_frames) / (max_in_frames-min_in_frames)) * np.iinfo(frames.dtype).max
    norm_frames = norm_frames.astype(frames.dtype)

    return norm_frames

# ## Function def PixelSignal(hroi_pixels)
def PixelSignal(hroi_pixels):
    """
        Extract individual pixel signal across all frames
    """
    pixel_signals = np.transpose(hroi_pixels, axes=[1, 0])

    # Remove empty signals
    pixel_signals = pixel_signals[~np.all(pixel_signals == 0, axis=1)]

    return(pixel_signals)

# Plots amplitudes for each frequency on x axis, pixels on y axis.
# 2D plot of frequencies detected in the region.
def plot_frequencies_2d(amplitudes, bins, outdir):

    # Ensures parameters are numpy arrays - meshgrid function works
    amplitudes = np.array(amplitudes)
    bins = np.array(bins)

    # 3D Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = bins
    y = range(len(amplitudes))
    X, Y = np.meshgrid(x, y)

    ax.plot_surface(X, Y, amplitudes)
    ax.set_xlabel('frequency')
    ax.set_ylabel('pixel')
    ax.set_zlabel('frequency amplitude')

    out_fig = os.path.join(outdir, "fourier_signals_3D.png")
    plt.savefig(out_fig, bbox_inches='tight')

    plt.close()

    # 2D heatmap (adapted from Erasmus Cedernaes, stackoverflow)
    fig, ax = plt.subplots()

    c = ax.pcolormesh(X, Y, amplitudes, cmap='hot', vmin=0, vmax=np.max(amplitudes))

    # set the limits of the plot to the limits of the data
    ax.axis([X.min(), X.max(), Y.min(), Y.max()])

    ax.set_title('Frequency intensities per pixel')
    ax.set_xlabel('frequency')
    ax.set_ylabel('pixel')
    fig.colorbar(c, ax=ax)

    out_fig = os.path.join(outdir, "fourier_signals_heatmap.png")
    plt.savefig(out_fig, bbox_inches='tight')

    plt.close()

def fourier_transform(hroi_pixels, times):
    amplitudes = []

    # Rotates array. Instead of frames, the first dimension are the pixels.
    pixel_signals = PixelSignal(hroi_pixels)

    # Get Discrete Fourier frequencies. Defined by sample length
    N = pixel_signals[0].size
    timestep = np.mean(np.diff(times))
    freqs = np.fft.rfftfreq(N, d=timestep)

    for pixel_signal in pixel_signals:

        # smoothe signal
        pixel_signal = savgol_filter(pixel_signal, window_length=5, polyorder=3)

        # Subtract any linear trends
        # Increases classification rate (depending on data, 1-12%)
        pixel_signal = detrend(pixel_signal)

        # Fast fourier transform
        fourier = np.fft.rfft(pixel_signal)

        # Power Spectral Density
        psd = np.abs(fourier) ** 2

        amplitudes.append(psd)

    return amplitudes, freqs

# Check if frequency specturm of pixel fullfills conditions to be viable
def freq_filter(pixel_freqs, freq_idx, min_snr, min_intensity):
    return (pixel_freqs[freq_idx] > min_intensity) and (pixel_freqs[freq_idx]/sum(pixel_freqs) > min_snr)

def analyse_frequencies(amplitudes, freqs):
    qc_data = {}
    bpm     = None

    # Get top frequency in each pixel
    max_indices = [np.argmax(pixel_freqs) for pixel_freqs in amplitudes]
    highest_freqs = [freqs[idx] for idx in max_indices]

    # SNR = Amplitude of max freq / sum of all freqs
    # Intensity = amplitude of top frequency
    SNR         = [(pixel_freqs[idx]/sum(pixel_freqs)) for pixel_freqs, idx in zip(amplitudes, max_indices)]
    intensity   = [(pixel_freqs[idx]) for pixel_freqs, idx in zip(amplitudes, max_indices)]

    # Take frequency that is most often the max
    max_freq = statistics.mode(highest_freqs)

    ### LOOK AT HARMONICS OF HIGHEST FREQUENCY

    # # Add max freq and harmonics to candidae freqs
    # freq_step = freqs[1] - freqs[0]
    # lower_harmonic = freqs[np.where(np.abs(freqs-(max_freq/2)) < (freq_step/2))]
    # upper_harmonic = freqs[np.where(np.abs(freqs-(max_freq*2)) < (freq_step/2))]

    # candidate_freqs = np.concatenate(([max_freq], lower_harmonic, upper_harmonic))
    # candidate_idcs  = [np.where(freqs == freq) for freq in candidate_freqs]

    # # Pick the freq with strongest signal
    # intensities = []
    # pixel_count = []
    # for freq, idx in zip(candidate_freqs, candidate_idcs):

    #     # Filter by intensity and snr.
    #     freq_amplitudes = [pixel_freqs for pixel_freqs in amplitudes if freq_filter(pixel_freqs, idx, min_snr, min_intensity)]

    #     # Check if viable
    #     if not freq_amplitudes:
    #         intensities.append(None)
    #         pixel_count.append(None)
    #         continue

    #     i = np.average([pixel_freqs[idx] for pixel_freqs in freq_amplitudes])

    #     intensities.append(i)
    #     pixel_count.append(len(freq_amplitudes))

    # # Return if none satisfy qc control
    # if intensities.count(None) == len(intensities):
    #     return bpm, qc_data

    # # conversion necessar to avoid problems with None
    # max_idx = np.nanargmax(np.array(intensities, dtype=float))

    # bpm = candidate_freqs[max_idx] * 60
    # qc_data["Intensity"]        = intensities[max_idx]
    # qc_data["Viable pixels"]    = pixel_count[max_idx]
    # qc_data["Viability rate"]   = pixel_count[max_idx] / len(amplitudes)

    ### Intensity of harmonics?
    freq_step = freqs[1] - freqs[0]
    lower_harmonic = freqs[np.where(np.abs(freqs-(max_freq/2)) < (freq_step/2))]
    upper_harmonic = freqs[np.where(np.abs(freqs-(max_freq*2)) < (freq_step/2))]

    candidate_freqs = np.concatenate((lower_harmonic, upper_harmonic))
    candidate_idcs  = [np.where(freqs == freq) for freq in candidate_freqs]

    # Find highest top 5% intesity of harmonic. (higher or lower)
    harmonic_intensity = 0
    for freq, idx in zip(candidate_freqs, candidate_idcs):

        # Filter by intensity and snr.
        freq_amplitudes = [pixel_freqs[idx] for pixel_freqs in amplitudes]
        freq_amplitudes.sort()

        i = np.average(freq_amplitudes[-(math.ceil(len(freq_amplitudes)/20)):])

        if i > harmonic_intensity:
            harmonic_intensity = i

    qc_data["Harmonic Intensity"] = str(harmonic_intensity)

    ### PICK HIGHEST FREQUENCY - QC FILTER FOR MIN INTENSITY AND 
    # Get SNR of pixels which contained max_freq
    SNR         = [snr  for freq, snr   in zip(highest_freqs, SNR)          if freq == max_freq]
    intensity   = [i    for freq, i     in zip(highest_freqs, intensity)    if freq == max_freq]
    
    overall_snr = sum(SNR)/len(SNR)
    overall_i   = sum(intensity)/len(intensity)

    # top contributers SNR
    n = math.ceil(len(SNR)/20)
    SNR.sort()
    top_snr = np.average(SNR[-n:])

    # top contributers Signal Intensities
    intensity.sort()
    top_i = np.average(intensity[-n:])

    bpm = round(max_freq * 60)
    
    qc_data['SNR']              = overall_snr
    qc_data['Signal intensity'] = round(overall_i)

    qc_data['SNR Top 5%'] = top_snr
    qc_data['Signal Intensity Top 5%'] = round(top_i)

    signal_prominence = len(SNR)/len(highest_freqs)
    qc_data['Signal regional prominence'] = signal_prominence

    qc_data['Intensity/Harmonic Intensity (top 5 %)'] = top_i / harmonic_intensity
 
    #### Decisions from empirical analysis:
    if signal_prominence < 0.33:
        LOGGER.info("Failed frequency analysis: Common top frequency in less then 33% of pixels")
        bpm = None
    if (top_i/harmonic_intensity) < 4.8:
        LOGGER.info("Failed frequency analysis: Intense Harmonics present.")
        bpm = None
    if top_snr < 0.3:
        LOGGER.info("Failed frequency analysis: Noisy Frequency spectrum")
        bpm = None
    if top_i < 30000:
        LOGGER.info("Failed frequency analysis: Signal not strong enough.")
        bpm = None
    
    return bpm, qc_data

# WIP: new_fourier(), but using old_fourier_restructured as basis
def bpm_from_heartregion(hroi_pixels, times, out_dir):
    minBPM = 15
    maxBPM = 300

    # Get Frequency Spectrum for each pixel.
    amplitudes, freqs = fourier_transform(hroi_pixels, times)

    # Limit to frequencies within defined borders
    heart_freqs_indices = np.where(np.logical_and(freqs >= (minBPM/60), freqs <= (maxBPM/60)))[0]
    freqs       = freqs[heart_freqs_indices]
    amplitudes  = [pixel_freqs[heart_freqs_indices] for pixel_freqs in amplitudes]

    # Plot the pixel amplitudes as 
    plot_frequencies_2d(amplitudes, freqs, out_dir)

    # Attempt to find bpm
    bpm, qc_data = analyse_frequencies(amplitudes, freqs)

    if bpm is not None:
        bpm = np.around(bpm, decimals=2)

    return bpm, qc_data

# Sort and remove duplicate frames
def sort_frames(video, timestamps):
    timestamps_sorted, idcs = np.unique(timestamps, return_index=True)
    video_sorted            = video[idcs]

    return video_sorted, timestamps_sorted

# Calculate fps from first and last timestamp or use predefined value
def determine_fps(timestamps, fps_console_parameter):
    fps = 0

    # Determine FPS
    # Determine frame rate from time-stamps if unspecified
    if not fps_console_parameter:
        # total time acquiring frames in seconds
        timestamp0 = int(timestamps[0])
        timestamp_final = int(timestamps[-1])
        total_time = (timestamp_final - timestamp0) / 1000
        fps = len(timestamps) / total_time
        LOGGER.info("Calculated fps: " + str(round(fps, 2)))
    else:
        fps = fps_console_parameter
        LOGGER.info("Defined fps: " + str(round(fps, 2)))

    fps = round(fps, 2)
    return fps

# TODO: Fourier Transform expects equally spaced samples. Do cubicspline over timestamps and intepolate over missing values
# timestamp spacing can vary by a few ms. Provides equally spaced timestamps
def equally_spaced_timestamps(nr_of_frames, fps):
    frame2frame = 1/fps
        
    final_time = frame2frame * nr_of_frames
    equal_space_times = np.linspace(start=0, stop=final_time, num=nr_of_frames, endpoint=False)

    return equal_space_times

def absdiff_between_frames(video):
    # Last frame has no differencs
    frame2frame_changes = np.zeros_like(video[:-1])

    # Blur smooths noise -> focus on larger regional changes
    blurred_video = np.array([cv2.GaussianBlur(frame, (9, 9), 0) for frame in video])

    frame2frame_changes = np.array([cv2.absdiff(frame, blurred_video[i+1]) for i, frame in enumerate(blurred_video[:-1])])

    return frame2frame_changes


def threshold_changes(frame2frame_difference, min_area=300):
    # Only pixels with the most changes
    thresholded_differences = np.array([diff > threshold_triangle(diff) for diff in frame2frame_difference], dtype=np.uint8)

    # Opening to remove noise
    thresholded_differences = np.array([cv2.morphologyEx(diff, cv2.MORPH_OPEN, KERNEL) for diff in thresholded_differences])
    
    # Keep intensity of changes
    #thresholded_differences = np.multiply(thresholded_differences, frame2frame_difference)

    return thresholded_differences

def detect_movement(frame2frame_changes):
    stop_frame = len(frame2frame_changes)
    max_change = 0
    for i, frame in enumerate(frame2frame_changes):
        change = np.sum(frame)
        if change > max_change:
            max_change = change
            
        if change > 50000:
            stop_frame = i
            break
    
    return (stop_frame + 1), max_change

# Filter regions of interest by size and select region with most overlap
def hroi_from_blobs(regions_of_interest, most_changing_pixels_mask, min_area=300):
    ### Find contours, filter by size
    # Contour mask
    contours, _ = cv2.findContours(regions_of_interest, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Filter based on contour area
    candidate_contours = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            candidate_contours.append(contour)

    if len(candidate_contours) < 1:
        raise ValueError("Couldn't find a suitable heart region")

    hroi_mask = np.zeros_like(regions_of_interest)
    max_pixel_ratio = 0
    heart_contour = None
    for contour in candidate_contours:
        region_mask = np.zeros_like(regions_of_interest)
        region_mask = cv2.drawContours(region_mask, [contour], -1, 1, thickness=-1)

        overlap = np.logical_and(region_mask, most_changing_pixels_mask)
        overlap_pixels  = overlap.sum()
        pixel_ratio     = overlap_pixels / region_mask.sum()

        if pixel_ratio > max_pixel_ratio:
            max_pixel_ratio = pixel_ratio
            heart_contour = contour

    # Draw heart contour onto mask
    cv2.drawContours(hroi_mask, [heart_contour], -1, 1, thickness=-1)

    return hroi_mask

# hroi... heart region of interest
def HROI2(frame2frame_changes, min_area=300):
    # TODO: a lot of resolution is lost with the conversion. to uint16
    total_changes = np.sum(frame2frame_changes, axis=0, dtype=np.uint32)
    #total_changes = (total_changes/65535).astype(np.uint16)

    ### Create mask with most changing pixels
    nr_of_pixels_considered = 250
    top_changing_indices = np.unravel_index(np.argsort(total_changes.ravel())[-nr_of_pixels_considered:], 
                                            total_changes.shape)
    
    top_changing_mask = np.zeros((total_changes.shape), dtype=bool)

    # Label pixels based on based on the top changeable pixels
    top_changing_mask[top_changing_indices] = 1

    ### Threshold heart RoI to find regions
    all_roi = total_changes > threshold_yen(total_changes)
    all_roi = all_roi.astype(np.uint8)

    # Fill holes in blobs
    all_roi = cv2.morphologyEx(all_roi, cv2.MORPH_CLOSE, KERNEL)

    hroi_mask = hroi_from_blobs(all_roi, top_changing_mask)

    return hroi_mask, all_roi, total_changes, top_changing_mask

def draw_heart_qc_plot(single_frame, abs_changes, all_roi, hroi_mask, top_changing_pixels, out_dir):
    label_top_changes = label(top_changing_pixels)
    
    hroi_mask = cv2.cvtColor(hroi_mask, cv2.COLOR_GRAY2RGB)
    all_roi = cv2.cvtColor(all_roi, cv2.COLOR_GRAY2RGB)

    hroi_mask = color.label2rgb(label_top_changes, image=hroi_mask,
                              alpha=0.7, bg_label=0, bg_color=None, colors=[(1, 0, 0)])

    all_roi = color.label2rgb(label_top_changes, image=all_roi,
                              alpha=0.7, bg_label=0, bg_color=None, colors=[(1, 0, 0)])

    # Prepare outfigure
    out_fig = os.path.join(out_dir, "embryo_heart_roi.png")

    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    # First frame
    ax[0, 0].imshow(single_frame, cmap='gray')
    ax[0, 0].set_title('Embryo', fontsize=10)
    ax[0, 0].axis('off')

    # Summed Absolute Difference between sequential frames
    ax[0, 1].imshow(abs_changes)
    ax[0, 1].set_title('Summed Absolute\nDifferences', fontsize=10)
    ax[0, 1].axis('off')

    # Thresholded Differences
    ax[1, 0].imshow(hroi_mask)
    ax[1, 0].set_title('Thresholded Absolute\nDifferences', fontsize=10)
    ax[1, 0].axis('off')

    # Overlap between filtered RoI mask and pixel maxima
    ax[1, 1].imshow(all_roi)
    ax[1, 1].set_title('RoI overlap with maxima', fontsize=10)
    ax[1, 1].axis('off')

    # Save Figure
    plt.savefig(out_fig, bbox_inches='tight')
    plt.show()
    plt.close()

def assert_8bit(video):
    if video.dtype == np.uint32:
        video = (video/65535).astype(np.uint16)
    if video.dtype == np.uint16:
        video = (video/255).astype(np.uint8)

    return video

def video_with_roi(normed_video, frame2frame_changes, hroi_mask):
    normed_video        = assert_8bit(normed_video)
    frame2frame_changes = assert_8bit(frame2frame_changes)

    changes_video       = normVideo(frame2frame_changes)
    
    roi_video           = np.array([cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) for frame in normed_video])
    
    # Color in changes
    roi_video[:-1,:,:,1] = np.array([cv2.add(frame[:,:,2], change_mask)
                            for frame, change_mask in zip(roi_video[:-1], changes_video)])
    
    # Draw outline of Heart ROI
    contours, _ = cv2.findContours(hroi_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    roi_video = np.array([cv2.drawContours(frame, contours, -1, 255, thickness=2)
                            for frame in roi_video])

    return roi_video

# run the algorithm on a well video
# TODO: Move data consistency check like duplicate frames, empty frames somewhere else before maybe.
def run(video, args, video_metadata):
    LOGGER.info("Starting algorithmic analysis")

    bpm = None
    qc_attributes = {}

    ################################################################################ Setup
    # Add well position to output directory path
    out_dir = os.path.join( args['outdir'], 
                            video_metadata['channel'],
                            video_metadata['loop'] + '-' + video_metadata['well_id'])

    os.makedirs(out_dir, exist_ok=True)

    # Ensures np array not lists.
    video = np.asarray(video)
    timestamps = np.asarray(video_metadata['timestamps'])

    video, timestamps = sort_frames(video, video_metadata['timestamps'])
    fps = determine_fps(timestamps, args['fps'])

    timestamps = equally_spaced_timestamps(len(timestamps), fps)

    ################################# Normalize Frames
    LOGGER.info("Normalizing frames")
    save_video(video, fps, out_dir, "before_norm.mp4")
    # Normalize frames
    normed_video = normVideo(video)
    del video

    LOGGER.info("Writing video")
    save_video(normed_video, fps, out_dir, "embryo.mp4")

    ################################ Detect HROI and write into figure. 
    LOGGER.info("Detecting HROI")
    # Runs the region detection in 8 bit (No effect if video loaded in 8bit anyway)
    video8  = assert_8bit(normed_video)
    
    frame2frame_changes = absdiff_between_frames(video8)
    frame2frame_changes_thresh= threshold_changes(frame2frame_changes)

    # Detect movement and stop analysis early
    stop_frame, max_change = detect_movement(frame2frame_changes_thresh)
    qc_attributes["Stop frame"] = str(stop_frame)
    qc_attributes["Movement detection max"] = max_change

    # Break condition
    if stop_frame < 3*fps:
        LOGGER.info("Movement before 3 seconds. Stopping analysis")
        return None, fps, qc_attributes

    # Shorten videos
    normed_video                = normed_video[:stop_frame]
    video8                      = video8[:stop_frame]
    frame2frame_changes         = frame2frame_changes[:stop_frame]
    frame2frame_changes_thresh  = frame2frame_changes_thresh[:stop_frame]

    try:
        hroi_mask, all_roi, total_changes, top_changing_pixels = HROI2(frame2frame_changes_thresh)
    except ValueError as e: #TODO: create a cutom exception to avoid catching any system errors.
        LOGGER.info("Couldn't detect a suitable heart region")
        return None, fps, qc_attributes

    draw_heart_qc_plot( video8[0],
                        total_changes,
                        hroi_mask*255, 
                        all_roi*255, 
                        top_changing_pixels, 
                        out_dir)

    roi_qc_video = video_with_roi(video8, frame2frame_changes_thresh, hroi_mask)
    save_video(roi_qc_video, fps, out_dir, "embryo_changes.mp4")

    ################################ Keep only pixels in HROI
    # Blur the image before frequency analysis - reduces number of false positives (BPM assigned where no heart present)
    masked_greys = [cv2.GaussianBlur(frame, (9, 9), 20) for frame in normed_video]
    
    # delete pixels outside of mask (=HROI)
    # flattens frames to 1D arrays (following pixelwise analysis doesn't need to preserve shape of individual images)
    mask = np.invert(hroi_mask*255)
    hroi_pixels = np.asarray([np.ma.masked_array(frame, mask).compressed() for frame in normed_video])

    heart_size = np.size(hroi_pixels, 1)
    qc_attributes["Heart size"] = str(heart_size)

    # TODO: Maybe limit on selected pixels in fourier analysis -> bit tricky, need to pull out that info and map afterwards
    qc_attributes["HROI Change Intensity"] = str(np.sum(np.multiply(hroi_mask, np.sum(frame2frame_changes, axis=0))) / heart_size)

    empty_frames = [i for i, frame in enumerate(normed_video) if not np.any(cv2.bitwise_and(frame, frame, mask=hroi_mask))]
    qc_attributes["empty frames"] = str(len(empty_frames))

    ################################################################################ Fourier Frequency estimation
    LOGGER.info("Fourier frequency evaluation")
    
    # Run normally, Fourier in segemented area
    if not args['slowmode']:
        bpm, qc_data = bpm_from_heartregion(hroi_pixels, timestamps, out_dir)

    qc_attributes.update(qc_data)

    if not bpm:
        LOGGER.info("No bpm detected")

    plt.close('all') # fixed memory leak
    return bpm, fps, qc_attributes