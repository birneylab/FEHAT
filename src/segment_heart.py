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
import math
from matplotlib import pyplot as plt
import os
import logging

from statistics import mean

import numpy as np
import cv2

# import skimage
from skimage.filters import threshold_triangle, threshold_yen
from skimage.measure import label
from skimage import color

import scipy.stats
import scipy.interpolate
from scipy.signal import savgol_filter, detrend 

import matplotlib
from mpl_toolkits.mplot3d import axes3d
matplotlib.use('Agg')

# Read config
import configparser

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_dir, 'config.ini')

config = configparser.ConfigParser()
config.read(config_path)

################################################################################
##########################
##  Globals   ##
##########################

LOGGER = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# Kernel for image smoothing
KERNEL = np.ones((5, 5), np.uint8)

def save_video(video, fps, outdir, filename):
    """
        Main Algorithm
        Heart Segmentation and frequeny detection
    """
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

def normVideo(frames):
    """
        Normalise across frames to harmonise intensities
        TODO: A single outlier will worsen the normalization. Ignore extreme outliers for the stretching.
    """
    min_in_frames = np.min(frames)
    max_in_frames = np.max(frames)

    norm_frames = (np.subtract(frames, min_in_frames) / (max_in_frames-min_in_frames)) * np.iinfo(frames.dtype).max
    norm_frames = norm_frames.astype(frames.dtype)

    return norm_frames

def PixelSignal(hroi_pixels):
    """
        Extract individual pixel signal across all frames
    """

    # Prevent empty signals
    hroi_pixels[0] += 1

    pixel_signals = np.transpose(hroi_pixels, axes=[1, 0])

    return(pixel_signals)


def plot_frequencies_2d(amplitudes, bins, outdir):
    """
        Plots amplitudes for each frequency on x axis, pixels on y axis.
        2D plot of frequencies detected in the region.
    """

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

    c = ax.pcolormesh(X, Y, amplitudes, cmap='hot', shading='auto', vmin=0, vmax=np.max(amplitudes))

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

    # The following operations are performed on each pixel's individual brightness value in each frame
    # Remove outliers
    clean_signal = savgol_filter(pixel_signals, axis=1, window_length=5, polyorder=3)

    # Adjust for any potential linear trend
    clean_signal = detrend(clean_signal, axis=1)

    # Amplitudes for each pixel via FFT
    fft_return = np.fft.rfftn(clean_signal, axes=(1,))
    fft_return = fft_return / N                         # normalize intensity for arbitrary number of frames available.
    amplitudes = np.square(np.abs(fft_return))

    return amplitudes, freqs

def analyse_frequencies(amplitudes, freqs):
    bpm = None

    # Get top frequency in each pixel
    max_indices = [np.argmax(pixel_freqs) for pixel_freqs in amplitudes]
    highest_freqs = [freqs[idx] for idx in max_indices]
    
    # Take frequency that is most often the max
    max_freq = scipy.stats.mode(highest_freqs).mode[0]

    #pick highest frequency
    bpm = round(max_freq * 60)

    qc_data = frequency_qc_attributes(max_freq, freqs, amplitudes, max_indices, highest_freqs)

    return bpm, qc_data

def frequency_qc_attributes(max_freq, freqs, amplitudes, max_indices, highest_freqs):
    """
        Get qc attributes out of the frequency spectrum data
    """
    qc_data = {}

    ### INTENSITY OF HARMONICS
    # Find upper and lower harmonic
    freq_step = freqs[1] - freqs[0]
    lower_harmonic = freqs[np.where(np.abs(freqs-(max_freq/2)) < (freq_step/1.5))]
    upper_harmonic = freqs[np.where(np.abs(freqs-(max_freq*2)) < (freq_step/1.5))]

    # Ensures harmonics outside of potential spectrum are not considered
    candidate_freqs = np.concatenate((lower_harmonic, upper_harmonic))
    candidate_idcs  = [np.where(freqs == freq) for freq in candidate_freqs]

    # Find highest top 5% intesity of harmonic. (higher or lower)
    harmonic_intensity = 0
    for freq_idx in candidate_idcs:

        # Get amplitudes of harmonic
        freq_amplitudes = [pixel_freqs[freq_idx] for pixel_freqs in amplitudes]
        freq_amplitudes.sort()

        # limit to top 5% of values
        n_5percent = math.ceil(len(freq_amplitudes)/20)
        freq_amplitudes = freq_amplitudes[-n_5percent:]

        i = np.average(freq_amplitudes)

        if i > harmonic_intensity:
            harmonic_intensity = i

    qc_data["Harmonic Intensity"] = str(harmonic_intensity)

    ### SIGNAL TO NOISE RATIO & INTENSITY#
    # Get SNR of pixels which contained max_freq
    SNR         = [(pixel_freqs[idx]/sum(pixel_freqs))  for pixel_freqs, idx in zip(amplitudes, max_indices) if freqs[idx] == max_freq]
    intensity   = [(pixel_freqs[idx])                   for pixel_freqs, idx in zip(amplitudes, max_indices) if freqs[idx] == max_freq]

    overall_snr = sum(SNR)/len(SNR)
    overall_i   = sum(intensity)/len(intensity)

    n_5percent = math.ceil(len(SNR)/20)

    # top contributers SNR
    SNR.sort()
    top_snr = np.average(SNR[-n_5percent:])

    # top contributers Signal Intensities
    intensity.sort()
    top_i = np.average(intensity[-n_5percent:])
    
    qc_data['SNR']              = overall_snr
    qc_data['Signal intensity'] = round(overall_i)

    qc_data['SNR Top 5%'] = top_snr
    qc_data['Signal Intensity Top 5%'] = round(top_i)

    signal_prominence = len(SNR)/len(highest_freqs)
    qc_data['Signal regional prominence'] = signal_prominence

    qc_data['Intensity/Harmonic Intensity (top 5 %)'] = top_i / harmonic_intensity
 
    # #### Decisions from empirical analysis:
    # if signal_prominence < 0.33:
    #     LOGGER.info("Failed frequency analysis: Common top frequency in less then 33% of pixels")
    #     qc_data['qc_error'] = "Signal prominence < 33%"
    #     #bpm = None
    # if (top_i/harmonic_intensity) < 4.8:
    #     LOGGER.info("Failed frequency analysis: Intense Harmonics present.")
    #     qc_data['qc_error'] = "Intense harmonics"
    #     #bpm = None
    # if top_snr < 0.3:
    #     LOGGER.info("Failed frequency analysis: Noisy Frequency spectrum")
    #     qc_data['qc_error'] = "Noisy frequency spectrum"
    #     #bpm = None
    # if top_i < 30000:
    #     LOGGER.info("Failed frequency analysis: Signal not strong enough.")
    #     qc_data['qc_error'] = "Signal intensity low"
    #     #bpm = None
    
    return qc_data

def bpm_from_heartregion(hroi_pixels, times, out_dir):
    minBPM = config['ANALYSIS'].getint('MINBPM')
    maxBPM = config['ANALYSIS'].getint('MAXBPM')

    # Get Frequency Spectrum for each pixel.
    amplitudes, freqs = fourier_transform(hroi_pixels, times)

    # Limit to frequencies within defined borders
    heart_freqs_indices = np.where(np.logical_and(freqs >= (minBPM/60), freqs <= (maxBPM/60)))[0]
    
    freqs       = freqs[heart_freqs_indices]
    amplitudes  = np.array([pixel_freqs[heart_freqs_indices] for pixel_freqs in amplitudes])

    # Plot pixel amplitudes for manual quality control
    plot_frequencies_2d(amplitudes, freqs, out_dir)

    # Attempt to find bpm
    bpm, qc_data = analyse_frequencies(amplitudes, freqs)

    if bpm is not None:
        bpm = np.around(bpm, decimals=2)

    return bpm, qc_data

def sort_frames(video, timestamps):
    """
        Sort and remove duplicate frames
    """
    timestamps_sorted, idcs = np.unique(timestamps, return_index=True)
    video_sorted            = video[idcs]

    timestamps_sorted = np.asarray(timestamps_sorted, dtype=np.uint64)

    return video_sorted, timestamps_sorted

def determine_fps(timestamps, fps_console_parameter):
    """
        Calculate fps from first and last timestamp or use predefined value
    """
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

def equally_spaced_timestamps(nr_of_frames, fps):
    """
        timestamp spacing can vary by a few ms. Provides artificial, equally spaced timestamps
    """
    frame2frame = 1/fps
        
    final_time = frame2frame * nr_of_frames
    equal_space_times = np.linspace(start=0, stop=final_time, num=nr_of_frames, endpoint=False)

    return equal_space_times

def interpolate_timestamps(video, timestamps):
    """
        timestamp spacing can vary by a few ms. Provides interpolated timestamps
    """
    LOGGER.info("Interpolating timestamps")

    # Calculate equaly spaced sample points
    equal_space_times = np.linspace(start=timestamps[0], stop=timestamps[-1], num=len(video), endpoint=True)
    
    # Interpolate pixel values of the video
    # Quite ressource intensive for full resolution images. (~16GB for 130*2048*2048).
    # Splitting into 10 subparts to mitigate this.
    interpolated_video = []
    for sub_arr in np.array_split(video, 10, axis=1):

        interpolated_arr = scipy.interpolate.interp1d(timestamps, sub_arr, axis=0, kind="cubic")(equal_space_times)
        interpolated_arr = np.clip(interpolated_arr, 0, np.iinfo(video.dtype).max)
        interpolated_arr = np.asarray(interpolated_arr, dtype=video.dtype)

        interpolated_video.append(interpolated_arr)

    interpolated_video = np.concatenate(interpolated_video, axis=1)

    return interpolated_video, equal_space_times

def timestamps_in_seconds(timestamps):
    timestamps = np.asarray((timestamps - timestamps[0]) / 1000, dtype=np.float16)
    
    return timestamps

def absdiff_between_frames(video):
    """
        Frame to frame absolute difference
    """
    # Last frame has no differencs
    frame2frame_changes = np.zeros_like(video[:-1])

    # Blur smooths noise -> focus on larger regional changes
    blurred_video = np.array([cv2.GaussianBlur(frame, (9, 9), 0) for frame in video])

    frame2frame_changes = np.array([cv2.absdiff(frame, blurred_video[i+1]) for i, frame in enumerate(blurred_video[:-1])])

    return frame2frame_changes

def threshold_changes(frame2frame_difference, min_area=300):
    """
        Filter away pixels that change not much
    """
    # Only pixels with the most changes
    thresholded_differences = np.array([diff > threshold_triangle(diff) for diff in frame2frame_difference], dtype=np.uint8)

    # Opening to remove noise
    thresholded_differences = np.array([cv2.morphologyEx(diff, cv2.MORPH_OPEN, KERNEL) for diff in thresholded_differences])

    return thresholded_differences

def detect_movement(frame2frame_changes):
    """
        Detects movement based on absolute frame change threshold. Threshold found empirically.
        Returns start and stop frame with maximal uninterrupted video length
        
        warning: return value of 'stop_frame' is a slice index. When used as array index may yield out of bounds error.
    """
    max_change = 0
    movement_frames = []

    for i, frame in enumerate(frame2frame_changes):
        change = np.sum(frame)
        if change > max_change:
            max_change = change
            
        if change > 50000:
            movement_frames.append(i)
    
    # Pick longest sequence, in case of movement
    if movement_frames:
        # add first and last frame to tart/stop frame candidates
        movement_frames.insert(0,0)
        movement_frames.append(len(frame2frame_changes))
        max_length = 0

        for i in range(len(movement_frames)-1):
            length = movement_frames[i+1] - movement_frames[i]
            if length > max_length:
                start_frame = movement_frames[i]
                stop_frame  = movement_frames[i+1]
                max_length = length
    else:
        start_frame = 0
        stop_frame = len(frame2frame_changes)
    
    return start_frame, stop_frame, max_change

def hroi_from_blobs(regions_of_interest, min_area=300):
    """
        Get largest region
    """
    ### Find contours, filter by size
    # Contour mask
    contours, _ = cv2.findContours(regions_of_interest, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Filter based on contour area
    candidate_contours = []
    candidate_areas = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            candidate_contours.append(contour)
            candidate_areas.append(area)

    if len(candidate_contours) < 1:
        return None

    largest_region_idx = np.argmax(candidate_areas)
    heart_contour = candidate_contours[largest_region_idx]

    # Draw heart contour onto mask
    hroi_mask = np.zeros_like(regions_of_interest)
    cv2.drawContours(hroi_mask, [heart_contour], -1, 1, thickness=-1)

    return hroi_mask

def HROI(video, frame2frame_changes, timestamps):
    """
        hroi... heart region of interest
        Analyse frame2frame changes for heart region (also returns candidate pixels from intermediate steps)
    """

    minBPM = config['ANALYSIS'].getint('MINBPM')
    maxBPM = config['ANALYSIS'].getint('MAXBPM')

    # Create mask of all pixels that exhibited change
    change_mask = np.zeros_like(video[0])
    for frame in frame2frame_changes:
        change_mask = cv2.bitwise_or(frame, change_mask)

    # save indices(x,y coords) to later filter by snr
    indices = np.where(change_mask)

    # Extract changing pixels
    # change_pixels.shape = (nr_frames, nr_change_pixels)
    change_pixels = np.array([frame[indices] for frame in video])

    pixel_amplitudes, freqs = fourier_transform(change_pixels, timestamps)

    # Limit to frequencies within defined borders
    heart_freqs_indices = np.where(np.logical_and(freqs >= (minBPM/60), freqs <= (maxBPM/60)))[0]
    pixel_amplitudes  = np.array([pixel_freqs[heart_freqs_indices] for pixel_freqs in pixel_amplitudes])
    
    # Get SNR
    max_indices = [np.argmax(pix_amps)      for pix_amps in pixel_amplitudes]

    SNR = [(pix_amps[idx]/sum(pix_amps))    for pix_amps, idx in zip(pixel_amplitudes, max_indices)]
    SNR = np.array(SNR)

    # intensity = [(pix_amps[idx])           for pix_amps, idx in zip(pixel_amplitudes, max_indices)]
    # intensity = np.array(intensity)

    ###### Only pixels are considered that exhibit periodic change.
    ###### This is defined by having a clear peak in the frequency spectrum of the pixel signal.

    # Filter for SNR. (0.3 found empirically to be reasonable)
    # It would make sense to also filter by intensity, however even  very low value of 1.0 worsened results
    candidates  = np.where((SNR > 0.3)) # & (intensity > 1.0))

    # Assemble regions of interest with clear signal pixels
    candidates = (indices[0][candidates], indices[1][candidates])
    all_roi = np.zeros_like(video[0])
    all_roi[candidates] = 1

    # Fill holes
    all_roi = cv2.morphologyEx(all_roi, cv2.MORPH_CLOSE, KERNEL, iterations=1)

    # Consider all connected regions. Select most probably heart region.
    hroi_mask = hroi_from_blobs(all_roi)

    return hroi_mask, all_roi, change_mask

def save_image(image, name, outdir):

    # Prepare outfigure
    out_fig = os.path.join(outdir, name + ".png")

    fig, ax = plt.subplots()

    # First frame
    ax.imshow(image, interpolation='none')
    ax.set_title(name, fontsize=10)

    # Save Figure
    plt.savefig(out_fig, bbox_inches='tight')
    plt.close()


def draw_heart_qc_plot(single_frame, abs_changes, all_roi, hroi_mask, out_dir):

    # Prepare outfigure
    out_fig = os.path.join(out_dir, "embryo_heart_roi.png")

    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    # First frame
    ax[0, 0].imshow(single_frame, cmap='gray')
    ax[0, 0].set_title('Embryo', fontsize=10)
    ax[0, 0].axis('off')

    # Summed Absolute Difference between sequential frames
    ax[0, 1].imshow(abs_changes, interpolation='none')
    ax[0, 1].set_title('Pixels emitting change', fontsize=10)
    ax[0, 1].axis('off')

    # Thresholded Differences
    ax[1, 0].imshow(hroi_mask, interpolation='none')
    ax[1, 0].set_title('Emitting clear periodic change', fontsize=10)
    ax[1, 0].axis('off')

    # Overlap between filtered RoI mask and pixel maxima
    ax[1, 1].imshow(all_roi, interpolation='none')
    ax[1, 1].set_title('Largest of these regions', fontsize=10)
    ax[1, 1].axis('off')

    # Save Figure
    plt.savefig(out_fig, bbox_inches='tight')
    plt.close()

def assert_8bit(video):
    if video.dtype == np.uint32:
        video = (video/65535).astype(np.uint16)
    if video.dtype == np.uint16:
        video = (video/255).astype(np.uint8)

    return video

def video_with_roi(normed_video, frame2frame_changes, hroi_mask=None):
    """
        Draws a line around Region of interes in the video.
        Colors in frame2frame changes. Higher change -> stronger color
    """
    normed_video        = assert_8bit(normed_video)
    frame2frame_changes = assert_8bit(frame2frame_changes)

    changes_video       = normVideo(frame2frame_changes)
    
    roi_video           = np.array([cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) for frame in normed_video])

    # Increase brightness to contrast changes
    roi_video           = cv2.add(roi_video, 120)
    
    # Color in changes
    roi_video[:-1,:,:,1] = np.array([cv2.subtract(frame[:,:,1], change_mask)
                            for frame, change_mask in zip(roi_video[:-1], changes_video)])
    
    # Draw outline of Heart ROI
    if hroi_mask is not None:
        contours, _ = cv2.findContours(hroi_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        roi_video = np.array([cv2.drawContours(frame, contours, -1, 255, thickness=1)
                                for frame in roi_video])

    return roi_video

def run(video, args, video_metadata):
    """
        Main function
        Attempt to extract BPM from Medaka embryo video.

        returns bpm, fps and quality control values from various analyses steps.
    """
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
    timestamps = np.asarray(video_metadata['timestamps'], dtype=np.uint64)

    video, timestamps = sort_frames(video, video_metadata['timestamps'])
    fps = determine_fps(timestamps, args['fps'])

    ################################# Normalize Frames
    LOGGER.info("Normalizing frames")
    
    normed_video = normVideo(video)
    del video

    ################################# Interpolate pixel values (assume timestamps of filenames valid)
    artificial_timestamps = config['ANALYSIS'].getboolean('ARTIFICIAL_TIMESTAMPS')

    if artificial_timestamps:
        timestamps = equally_spaced_timestamps(len(timestamps), fps)
    else:
        normed_video, timestamps = interpolate_timestamps(normed_video, timestamps)
        timestamps = timestamps_in_seconds(timestamps)

    LOGGER.info("Writing video")
    save_video(normed_video, fps, out_dir, "embryo.mp4")

    ################################ Detect HROI and write into figure. 
    LOGGER.info("Detecting HROI")

    # Runs the region detection in 8 bit (No effect if video loaded in 8bit anyway)
    video8  = assert_8bit(normed_video)
    
    frame2frame_changes = absdiff_between_frames(video8)
    frame2frame_changes_thresh= threshold_changes(frame2frame_changes)

    # Detect movement and stop analysis early
    start_frame, stop_frame, max_change = detect_movement(frame2frame_changes_thresh)
    qc_attributes["Movement detection max"] = max_change
    qc_attributes["Start frame(movement)"] = str(start_frame)
    qc_attributes["Stop frame(movement)"] = str(stop_frame)

    # Break condition
    if (stop_frame - start_frame) < 4*fps:
        LOGGER.info("Can't find 4 second long clip without movement. Stopping analysis")
        return None, fps, qc_attributes

    # Adjust data for movement
    normed_video                = normed_video[start_frame:stop_frame]
    video8                      = video8[start_frame:stop_frame]
    frame2frame_changes         = frame2frame_changes[start_frame:stop_frame]
    frame2frame_changes_thresh  = frame2frame_changes_thresh[start_frame:stop_frame]
    timestamps                  = timestamps[start_frame:stop_frame]

    # Detect region of interest
    hroi_mask, all_roi, total_changes = HROI(normed_video, frame2frame_changes_thresh, timestamps)

    if hroi_mask is None:
        LOGGER.info("Couldn't detect a suitable heart region")

        # image of pixels considered for the heart region..
        save_image(all_roi*255, "ROI_pixels", out_dir)
        return None, fps, qc_attributes

    # Output video and region plot for manual quality control.
    roi_qc_video = video_with_roi(video8, frame2frame_changes, hroi_mask)
    save_video(roi_qc_video, fps, out_dir, "embryo_changes.mp4")

    draw_heart_qc_plot(video8[0],
                        total_changes,
                        hroi_mask*255, 
                        all_roi*255, 
                        out_dir)
                        
    ################################ Keep only pixels in HROI
    # delete pixels outside of mask (=HROI)
    # flattens frames to 1D arrays (following pixelwise analysis doesn't need to preserve shape of individual images)
    mask = np.invert(hroi_mask*255)
    hroi_pixels = np.asarray([np.ma.masked_array(frame, mask).compressed() for frame in normed_video])

    heart_size = np.size(hroi_pixels, 1)
    qc_attributes["Heart size"] = str(heart_size)

    # Sum of absolute brightness changes over all pixels, over all frames.
    qc_attributes["HROI Change Intensity"] = str(np.sum(np.multiply(hroi_mask, np.sum(frame2frame_changes, axis=0))) / heart_size)

    # For quality control. Fluorescend data may need adjustment for this.
    empty_frames = [i for i, frame in enumerate(normed_video) if not np.any(cv2.bitwise_and(frame, frame, mask=hroi_mask))]
    qc_attributes["empty frames"] = str(len(empty_frames))

    ################################################################################ Fourier Frequency estimation
    LOGGER.info("Fourier frequency evaluation")
    
    # Run normally, Fourier in segemented area
    if not args['slowmode']:
        bpm, qc_data = bpm_from_heartregion(hroi_pixels, timestamps, out_dir)

    # Add attributes from frequency analysis to dictionary
    qc_attributes.update(qc_data)

    if not bpm:
        LOGGER.info("No bpm detected")

    # ensures no memory leaks
    plt.close('all')
    
    return bpm, fps, qc_attributes