############################################################################################################
# Authors: 
#   Sebastian Stricker, Uni Heidelberg, sebastian.stricker@stud.uni-heidelberg.de
#   Marcio Ferreira,    EMBL-EBI,       marcio@ebi.ac.uk
# Date: 08/2021
# License: Contact authors
###
# Handles all I/O interactions. Reading images, creating result csv, analysisng input directory structures
###
############################################################################################################
import logging
import os
import numpy as np
import csv

import pathlib

import glob2
import cv2
from matplotlib import pyplot as plt

#from matplotlib import pyplot as plt

logging.getLogger('matplotlib.font_manager').disabled = True
LOGGER = logging.getLogger(__name__)

# Goes through all channels and loops and yields well data fields and paths to frames sorted by frame index.


def well_video_generator(indir, channels, loops):
    # -A001--PO01--LO001--CO1--SL001--PX32500--PW0070--IN0020--TM280--X014600--Y011401--Z214683--T0000000000--WE00001.tif

    # Grab all paths to all frames
    all_frames = glob2.glob(indir + '*.tif') + glob2.glob(indir + '*.tiff')

    # Channel
    for channel in channels:
        LOGGER.info("### CHANNEL " + channel + " ###")
        channel_videos = [frame for frame in all_frames if channel in frame]
        # Loop
        for loop in loops:
            LOGGER.info("### LOOP " + loop + " ###")
            loop_videos = [frame for frame in channel_videos if loop in frame]

            # Well
            for well_id in range(1, 97):
                well_id = ('WE00' + '{:03d}'.format(well_id))

                well_frames = [
                    frame for frame in loop_videos if well_id in frame]

                # Skipping costs (close to) no time and the logic is simpler than extracting the exact set of wells per loop/channel.
                # Could be solved cleaner though.
                if not well_frames:
                    continue

                # Sort frames in correct order
                frame_indices = [frameIdx(path) for path in well_frames]
                _, well_frames_sorted = (list(t) for t in zip(
                    *sorted(zip(frame_indices, well_frames))))

                metadata = {'well_id': well_id,
                            'loop': loop, 'channel': channel}
                yield well_frames_sorted, metadata


def detect_experiment_directories(indir):
    subdirs = set()
    subdir_list = {os.path.join(os.path.dirname(p), '') for p in glob2.glob(
        indir + '/*/')}  # set([os.path.dirname(p) for p in glob2.glob(indir + '/*/')])

    # Condition: Tiffs inside or croppedRAWTifffolder
    # The previous function seems not to work properly.
    for path in subdir_list:
        if os.path.basename(os.path.normpath(path)) == 'croppedRAWTiff':
            subdirs = [path]
            return subdirs

        cond_1 = os.path.isdir(os.path.join(path, "croppedRAWTiff"))
        cond_2 = glob2.glob(path + '*.tif') + glob2.glob(path + '*.tiff')
        if cond_1 or cond_2:
            subdirs.add(path)

    # No subdirectories. Add indir as only folder
    if not subdirs:
        subdirs.add(indir)

    return subdirs

# TODO: get default to greyscale, as the videos are only in greyscale, conversion everywhere is overhead


def load_well_video_8bits(frame_paths_sorted, max_frames=-1):
    LOGGER.info("Loading video as 8 bits")
    video8 = []
    for index, path in enumerate(frame_paths_sorted):
        # check if the function is supposed to read every frame (-1), otherwise, runs only the number of frames specified in max_frames argument
        # it is usefull as if we need crop and save the images, we only need the first 5 frames to make the average of position.
        if max_frames == -1 or index < max_frames:
            frame = cv2.imread(path, 1)  # 1 flag to read image as rgb
            video8.append(frame)
    return video8


def load_well_video_16bits(frame_paths_sorted):
    LOGGER.info("Loading video as 16 bits")
    video16 = []
    for path in frame_paths_sorted:
        # -1 flag to read image as it is (16 bits)
        frame = cv2.imread(path, -1)
        video16.append(frame)
    return video16


def extract_timestamps(sorted_frame_paths):
    # splits every path at '-T'. Picks first 10 chars of the string that starts with a number.
    timestamps = [[s for s in path.split(
        '-T') if s[0].isdigit()][-1][0:10] for path in sorted_frame_paths]
    return timestamps

# Get metadata about the directory that is read in
# Number of videos and channels and loops present.


def extract_data(indir):
    # -A001--PO01--LO001--CO1--SL001--PX32500--PW0070--IN0020--TM280--X014600--Y011401--Z214683--T0000000000--WE00001.tif
    LOGGER.info("### Extracting data from image names ###")

    # Grab first frame of all videos
    tiffs = glob2.glob(indir + '*SL001' + '*.tif') + \
        glob2.glob(indir + '*SL001' + '*.tiff')
    nr_of_videos = len(tiffs)

    if not tiffs:
        raise ValueError("Could not find any tiffs inside " + indir)

    # Extract different channels
    # using a set, gives only unique values
    channels = {'CO' + tiff.split('-CO')[-1][0] for tiff in tiffs}
    channels = sorted(list(channels))

    # Extract different Loops
    # using a set, gives only unique values
    loops = {'LO' + tiff.split('-LO')[-1][0:3] for tiff in tiffs}
    loops = sorted(list(loops))

    return nr_of_videos, channels, loops

# From Tim-script


def frameIdx(path):
    idx = path.split('-SL')[-1]
    idx = idx.split('-')[0]
    idx = int(idx)
    return idx


def well_video_exists(indir, channel, loop, well_id):
    all_frames = glob2.glob(indir + '*.tif') + glob2.glob(indir + '*.tiff')
    video_frames = [frame for frame in all_frames if (
        channel in frame and loop in frame and well_id in frame)]

    if video_frames:
        return True
    else:
        return False

# Results:
# Dictionary {'channel': [], 'loop': [], 'well': [], 'heartbeat': []}
# TODO: Transfer functionality into pandas dataframes. Probably more stable and clearer


def write_to_spreadsheet(outdir, results, experiment_id):
    LOGGER.info("Saving acquired data to spreadsheet")
    outfile_name = "results_" + experiment_id + ".csv"
    outpath = os.path.join(outdir, outfile_name)

    # Don't erase previous results by accident
    if os.path.isfile(outpath):
        LOGGER.warning(
            "Outdir already contains results file. Writing new file version")

    version = 2
    while os.path.isfile(outpath):
        outpath = os.path.join(outdir, "results_v" + str(version) + ".csv")
        version += 1

    # Write results in file
    with open(outpath, 'w') as outfile:
        writer = csv.writer(outfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)

        writer.writerow(['Index', 'WellID', 'Loop',
                        'Channel', 'Heartrate (BPM)'])
        nr_of_results = len(results['heartbeat'])
        for idx in range(nr_of_results):
            ch = results['channel'][idx]
            loop = results['loop'][idx]
            well = results['well'][idx]
            bpm = results['heartbeat'][idx]
            if bpm is None:
                bpm = "NA"
            else:
                bpm = str(bpm)
            writer.writerow([str(idx+1), well, loop, ch, bpm])


def save_cropped(cut_images, args, images_path):
    # function to save the cropped images
    os.makedirs(os.path.join(
        args.outdir, 'cropped_by_EBI_script/'), exist_ok=True)
    for index, img in enumerate(cut_images):
        final_part_path = pathlib.PurePath(images_path[0]).name
        outfile_path = os.path.join(
            args.outdir, 'cropped_by_EBI_script/', final_part_path)
        # write the image
        cv2.imwrite(outfile_path, img)
        # get first image for saving as image offset
        if index == 0:  # avoid plot more than the first frame
            outfile_path = os.path.join(args.outdir, "offset_verifying.png")
            cv2.imwrite(outfile_path, img)

        # create a dictionary for the first cut image id it does not exist. If it exist, just append the cut image to the specific loop/channel.
        # it is necessary because we want to replot after each well, that is, to be able to skip the crop script but have the partial results plotted


def save_panel(resulting_dict_from_crop, args):

    # function used to create ans save the panel with cropped images
    for item in resulting_dict_from_crop.items():
        if "positions_" not in item[0]:
            axes = []  # will be used to plot the first image for each well bellow
            rows = 8
            cols = 12
            fig = plt.figure(figsize=(10, 8))
            suptitle = plt.suptitle(
                'General view of every cropped well in ' + item[0], y=1.01, fontsize=14, color='blue')
            counter = 1
            for cut_image, position in zip(item[1], resulting_dict_from_crop['positions_' + item[0]]):
                position_number = position[-2:]
                formated_counter = '{:02d}'.format(counter)
                while (position_number > formated_counter):  # do not save image
                    axes.append(fig.add_subplot(rows, cols, counter))
                    subplot_title = ("WE000" + str(formated_counter))
                    axes[-1].set_title(subplot_title,
                                       fontsize=11, color='blue')
                    plt.xticks([], [])
                    plt.yticks([], [])
                    plt.tight_layout()
                    # will not plot image but save figure anyway
                    plt.imshow(np.zeros((0, 0)))
                    outfile_path = os.path.join(
                        args.outdir, item[0] + "_panel.png")
                    counter += 1
                    formated_counter = '{:02d}'.format(counter)

                else:  # save image
                    axes.append(fig.add_subplot(rows, cols, counter))
                    subplot_title = (position)
                    axes[-1].set_title(subplot_title,
                                       fontsize=11, color='blue')
                    plt.xticks([], [])
                    plt.yticks([], [])
                    plt.tight_layout()
                    # plot in panel the last cropped image from the loop above
                    plt.imshow(cut_image)
                    # save figure
                    outfile_path = os.path.join(
                        args.outdir, item[0] + "_panel.png")
                    counter += 1
                    formated_counter = '{:02d}'.format(counter)

            print("counter")
            print(counter)

            while (counter < 97):  # do not save image
                axes.append(fig.add_subplot(rows, cols, counter))
                subplot_title = ("WE000" + str(formated_counter))
                axes[-1].set_title(subplot_title,
                                   fontsize=11, color='blue')
                plt.xticks([], [])
                plt.yticks([], [])
                plt.tight_layout()
                # will not plot image but save figure anyway
                plt.imshow(np.zeros((0, 0)))
                outfile_path = os.path.join(
                    args.outdir, item[0] + "_panel.png")
                counter += 1
                formated_counter = '{:02d}'.format(counter)

            plt.savefig(outfile_path, bbox_extra_artists=(
                        suptitle,), bbox_inches="tight")

# def save_cropped_img(outdir, img, well_id, loop_id):
#     name = loop_id + '-' + str(well_id)

#     out_fig = os.path.join(outdir, name + "_cropped.png")
#     plt.imshow(img)
#     plt.title('Original Frame', fontsize=10)
#     plt.axis('off')
#     plt.savefig(out_fig,bbox_inches='tight')
#     plt.close()

# def write_videos(path, videos, names, is_color=False):
#     # iterate over videos and corresponding well names in parallel
#     for video, name in zip(videos, names):
#         write_video(path, video, name, is_color)

# def write_video(path, video, name, is_color=False):
#     name = name + ".mp4"
#     LOGGER.debug("Writing video " + name)

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')

#     try:
#         height, width, layers = video[0].shape
#     except:
#         height, width = video[0].shape

#     size = (width,height)
#     out_vid = os.path.join(path, name)
#     out = cv2.VideoWriter(out_vid,fourcc, fps, size, is_color)
#     for i in range(len(video)):
#         out.write(video[i])
#     out.release()
