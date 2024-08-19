############################################################################################################
# Authors:
#   Marcio Ferreira,    EMBL-EBI,       marcio@ebi.ac.uk                            (Current Maintainer)
#   Sebastian Stricker, Uni Heidelberg, sebastian.stricker@stud.uni-heidelberg.de   (Current Maintainer)
# Date: 08/2021
# License: Contact authors
###
# Cropping script ported from Fiji to python.
# Work in progress.
###
############################################################################################################
import os
import logging

from statistics import mean

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

# import skimage
from skimage.filters import threshold_triangle

LOGGER = logging.getLogger(__name__)

def crop_border(img, ratio=0.15):
    height, width = img.shape[:2]

    # Calculate the cropping margins
    top = int(height * ratio)
    left = int(width * ratio)
    bottom = height - top
    right = width - left

    cropped_image = img[top:bottom, left:right]
    return cropped_image

def get_most_central_blobs(binary_img):

    inverted_img = cv2.bitwise_not(binary_img) - 254

    # Find all connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted_img)

    # Define the center of the image
    image_center = np.array(inverted_img.shape) // 2

    # Initialize variables to keep track of the most central blob
    min_distance = float('inf')
    central_blob_label = None

    # Loop through all connected components (ignore label 0 which is the background)
    for i in range(1, num_labels):
        if stats[i][0] == 0 or stats[i][1] == 0:
            # Quick pre-filter. Do not consider blobs, that touch the top or left side border of the image.
            continue

        # Compute the distance of the blob's centroid to the image center
        distance = np.linalg.norm(centroids[i] - image_center)
        
        # Update the most central blob if this one is closer
        if distance < min_distance:
            min_distance = distance
            central_blob_label = i

    # Create an output image with only the most central blob
    output_image = np.zeros_like(binary_img)
    if central_blob_label is not None:
        output_image[labels != central_blob_label] = 1
    return output_image, centroids[central_blob_label]

### Cropping Feature
# final_dist_graph(bpm_fourier)    ## debug
def embryo_detection(video, embryo_size=450, border_ratio=0.15):
    center_of_embryo_list = []
    for img, i in zip(video, range(5)):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # initial crop (due to imageJ macro file)
        img_gray = crop_border(img_gray, border_ratio)

        # norming
        cv2.normalize(img_gray, img_gray, 0, 255, cv2.NORM_MINMAX)

        img_blurred = gaussian_filter(img_gray, sigma=200)
        img_cleaned = (img_gray.astype(np.float32) / (img_blurred + 1e-6)) * 8000  # Avoid division by zero
        
        img_cleaned_blurred = gaussian_filter(img_cleaned, sigma=10)
        cv2.normalize(img_cleaned_blurred, img_cleaned_blurred, 0, 255, cv2.NORM_MINMAX)

        threshold = threshold_triangle(img_cleaned_blurred)
        thresh_img = (img_cleaned_blurred > threshold).astype(np.uint8)

        # fill in any holes
        thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

        # Filter out any remaining blobs on the border of the image.
        # Use blob center as result
        filtered_img, centroid = get_most_central_blobs(thresh_img)

        x_center, y_center = centroid

        height, width = img_gray.shape[:2]
        x_center += width * border_ratio
        y_center += height * border_ratio

        center_of_embryo_list.append((x_center, y_center))

    XY_average = (mean([i[0] for i in center_of_embryo_list]), mean(
        [i[1] for i in center_of_embryo_list]))

    return XY_average

def crop_2(video, embryo_size, embryo_coordinates, resulting_dict_from_crop, video_metadata):
    # avoid window size lower than 50 or higher than the minimum dimension of images
    # window size is the size of the window that the script will crop starting from centre os mass,
    # and can be passed as argument in command line (100 is default)
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
            x_lim = [int(embryo_coordinates[0])-embryo_size, int(embryo_coordinates[0])+embryo_size]
            y_lim = [int(embryo_coordinates[1])-embryo_size, int(embryo_coordinates[1])+embryo_size]
            cut_image = img[y_lim[0]: y_lim[1], x_lim[0]: x_lim[1]]
           
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
