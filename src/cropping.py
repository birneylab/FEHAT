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

# import skimage
from skimage.filters import threshold_yen

LOGGER = logging.getLogger(__name__)

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
