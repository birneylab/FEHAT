############################################################################################################
# Authors: 
#   Sebastian Stricker, Uni Heidelberg, sebastian.stricker@stud.uni-heidelberg.de
#   Marcio Ferreira,    EMBL-EBI,       marcio@ebi.ac.uk
# Date: 08/2021
# License: GNU GENERAL PUBLIC LICENSE Version 3
###
# Handles all I/O interactions. Reading images, creating result csv, analysisng input directory structures
###
############################################################################################################
import logging
from pathlib import Path
import pathlib
import pickle

import cv2
import numpy as np

from matplotlib import pyplot as plt

import configparser

# Read config
parent_dir = Path(__file__).resolve().parents[1]
config_path = parent_dir / 'config.ini'

config = configparser.ConfigParser()
config.read(config_path)

logging.getLogger('matplotlib.font_manager').disabled = True
LOGGER = logging.getLogger(__name__)

# Goes through all channels and loops and yields well data fields and paths to frames sorted by frame index.
def well_video_generator(indir, channels, loops):
    # -A001--PO01--LO001--CO1--SL001--PX32500--PW0070--IN0020--TM280--X014600--Y011401--Z214683--T0000000000--WE00001.tif

    # Grab all paths to all frames
    all_frames = list(indir.glob('*.tif')) + list(indir.glob('*.tiff'))

    # Channel
    for channel in channels:
        LOGGER.info("### CHANNEL " + channel + " ###")
        channel_videos = [frame for frame in all_frames if channel in frame.name]
        # Loop
        for loop in loops:
            LOGGER.info("### LOOP " + loop + " ###")
            loop_videos = [frame for frame in channel_videos if loop in frame.name]

            # Well
            for well_id in range(1, 97):
                well_id = ('WE00' + '{:03d}'.format(well_id))

                well_frames = [frame for frame in loop_videos if well_id in frame.name]

                # Skipping costs (close to) no time and the logic is simpler than extracting the exact set of wells per loop/channel.
                # Could be solved cleaner though.
                if not well_frames:
                    continue

                # Sort frames in correct order
                frame_indices = [frameIdx(path.name) for path in well_frames]
                _, well_frames_sorted = (list(t) for t in zip(*sorted(zip(frame_indices, well_frames))))

                metadata = {'well_id': well_id, 'loop': loop, 'channel': channel}
                yield well_frames_sorted, metadata


def detect_experiment_directories(indir):
    subdirs = set()
    subdir_list = indir.glob('*/')

    # Condition: Tiffs inside or croppedRAWTiff-folder present
    for subdir in subdir_list:

        cond_1 = (subdir / "croppedRAWTiff").is_dir()
        cond_2 = list(subdir.glob('*.tif')) + list(subdir.glob('*.tiff'))

        if cond_1 or cond_2:
            subdirs.add(subdir)

    # No subdirectories. Add indir as only folder
    if not subdirs:
        subdirs.add(indir)

    return subdirs

def load_decision_tree():
    tree_path = parent_dir / config['DEFAULT']['DECISION_TREE_PATH']
    trained_tree = None
                
    if not tree_path.exists():
        LOGGER.Warning("Trained model for qc analysis not found. Please train model first.")
        # TODO: Exit the qc_analysis if the trained tree is not saved.
    else:
        LOGGER.info("Trained model for qc analysis found. Proceeding with qc analysis.")
        with open(tree_path, 'rb') as f:
            trained_tree = pickle.load(f)

    return trained_tree

# imread_flag:
#   -1  - as is (greyscale 16bit)
#   0   - greyscale 8 bit
#   1   - color     8 bit
def load_video(frame_paths, imread_flag=0, max_frames=np.inf):
    LOGGER.info("Loading video...")

    test_frame = cv2.imread(frame_paths[0], imread_flag)

    video = np.empty(shape=(len(frame_paths), *test_frame.shape), dtype=test_frame.dtype)
    for i, path in enumerate(frame_paths):
        if i >= max_frames:
            break

        frame = cv2.imread(path, imread_flag)
        video[i] = frame
    return np.asarray(video)

def extract_timestamps(sorted_frame_paths):
    # splits every path at '-T'. Picks first 10 chars of the string that starts with a number.
    timestamps = [[s for s in path.name.split('-T') if s[0].isdigit()][-1][0:10] for path in sorted_frame_paths]
    return timestamps

# Get metadata about the directory that is read in
# Number of videos and channels and loops present.
def extract_data(indir):
    # -A001--PO01--LO001--CO1--SL001--PX32500--PW0070--IN0020--TM280--X014600--Y011401--Z214683--T0000000000--WE00001.tif
    LOGGER.info("### Extracting data from image names ###")

    # Grab first frame of all videos
    tiffs = list(indir.glob('*SL001*.tif')) + \
            list(indir.glob('*SL001*.tiff'))
    nr_of_videos = len(tiffs)

    if not tiffs:
        raise ValueError("Could not find any tiffs inside " + indir)

    # Extract different channels
    # using a set, gives only unique values
    channels = {'CO' + tiff.name.split('-CO')[-1][0] for tiff in tiffs}
    channels = sorted(list(channels))

    # Extract different Loops
    # using a set, gives only unique values
    loops = {'LO' + tiff.name.split('-LO')[-1][0:3] for tiff in tiffs}
    loops = sorted(list(loops))

    return nr_of_videos, channels, loops

# From Tim-script
def frameIdx(path):
    idx = path.split('-SL')[-1]
    idx = idx.split('-')[0]
    idx = int(idx)
    return idx

def well_video_exists(indir, channel, loop, well_id):
    all_frames = list(indir.glob('*.tif')) + list(indir.glob('*.tiff'))
    video_frames = [frame for frame in all_frames
                    if (channel in frame and 
                        loop    in frame and 
                        well_id in frame)]

    if video_frames:
        return True
    else:
        return False

# Results:
# Pandas df
#   columns: {'channel', 'loop', 'well_id', 'bpm', 'fps', ...qc_attributes}
def write_to_spreadsheet(outdir, results, experiment_id):
    LOGGER.info("Saving acquired data to spreadsheet")
    software_version = config['DEFAULT']['VERSION']
    outfile_name = f"results_{experiment_id}_{software_version}.csv"
    outpath = outdir / outfile_name

    # Don't erase previous results by accident
    if outpath.is_file():
        LOGGER.warning("Outdir already contains results file. Writing new file version")

    version = 2
    while outpath.is_file():
        outpath = outdir / f"results_{experiment_id}_{software_version}_{version}.csv"
        version += 1

    #header = ['Index', 'WellID', 'Well Name', 'Loop', 'Channel', 'Heartrate (BPM)', 'fps', 'version']
    results = results.rename(columns={  'well_id'   : 'WellID', 
                                        'loop'      : 'Loop', 
                                        'channel'   : 'Channel',
                                        'bpm'       : 'Heartrate (BPM)'})
    
    # Map Well ID to Well Name. Order of recording to Well-Plate layout.
    results.insert(loc=1, column='Well Name', value=results['WellID'].map(well_id_name_table))

    # Order columns before output
    ordered_cols = ['WellID', 'Well Name', 'Loop', 'Channel', 'Heartrate (BPM)', 'fps', 'version']
    ordered_cols += [col for col in results.columns if col not in ordered_cols] # Debug columns, if any
    results = results[ordered_cols]

    results.index += 1

    results.to_csv(outpath, index=True, index_label='Index', na_rep='NA')

def save_cropped(cut_images, args, images_path):
    outpath = args.outdir / 'croppedRAWTiff/'
    outpath.mkdir(parents=True, exist_ok=True)

    for index, img in enumerate(cut_images):
        final_part_path = pathlib.PurePath(images_path[index]).name
        outfile_path = outpath / final_part_path

        # write the image
        cv2.imwrite(outfile_path, img)

        # get first image for saving as image offset
        if index == 0:  # avoid plot more than the first frame
            outfile_path = args.outdir / "offset_verifying.png"
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
                outfile_path = args.outdir / f"{item[0]}_panel.png"
                counter += 1
                formated_counter = '{:02d}'.format(counter)        

            while (counter < 97):  # do not save image
                axes.append(fig.add_subplot(rows, cols, counter))
                subplot_title = ("WE000" + str(formated_counter))
                axes[-1].set_title(subplot_title,
                                   fontsize=11, color='blue')
                plt.xticks([], [])
                plt.yticks([], [])
                plt.tight_layout()
                outfile_path = args.outdir / f"{item[0]}_panel.png"
                counter += 1
                formated_counter = '{:02d}'.format(counter)

            plt.savefig(outfile_path, bbox_extra_artists=(
                        suptitle,), bbox_inches="tight")

well_id_name_table = {  'WE00001': 'A001', 'WE00002': 'A002', 'WE00003': 'A003', 'WE00004': 'A004', 'WE00005': 'A005', 'WE00006': 'A006', 
                        'WE00007': 'A007', 'WE00008': 'A008', 'WE00009': 'A009', 'WE00010': 'A010', 'WE00011': 'A011', 'WE00012': 'A012', 
                        'WE00013': 'B012', 'WE00014': 'B011', 'WE00015': 'B010', 'WE00016': 'B009', 'WE00017': 'B008', 'WE00018': 'B007', 
                        'WE00019': 'B006', 'WE00020': 'B005', 'WE00021': 'B004', 'WE00022': 'B003', 'WE00023': 'B002', 'WE00024': 'B001', 
                        'WE00025': 'C001', 'WE00026': 'C002', 'WE00027': 'C003', 'WE00028': 'C004', 'WE00029': 'C005', 'WE00030': 'C006', 
                        'WE00031': 'C007', 'WE00032': 'C008', 'WE00033': 'C009', 'WE00034': 'C010', 'WE00035': 'C011', 'WE00036': 'C012', 
                        'WE00037': 'D012', 'WE00038': 'D011', 'WE00039': 'D010', 'WE00040': 'D009', 'WE00041': 'D008', 'WE00042': 'D007', 
                        'WE00043': 'D006', 'WE00044': 'D005', 'WE00045': 'D004', 'WE00046': 'D003', 'WE00047': 'D002', 'WE00048': 'D001', 
                        'WE00049': 'E001', 'WE00050': 'E002', 'WE00051': 'E003', 'WE00052': 'E004', 'WE00053': 'E005', 'WE00054': 'E006', 
                        'WE00055': 'E007', 'WE00056': 'E008', 'WE00057': 'E009', 'WE00058': 'E010', 'WE00059': 'E011', 'WE00060': 'E012', 
                        'WE00061': 'F012', 'WE00062': 'F011', 'WE00063': 'F010', 'WE00064': 'F009', 'WE00065': 'F008', 'WE00066': 'F007', 
                        'WE00067': 'F006', 'WE00068': 'F005', 'WE00069': 'F004', 'WE00070': 'F003', 'WE00071': 'F002', 'WE00072': 'F001', 
                        'WE00073': 'G001', 'WE00074': 'G002', 'WE00075': 'G003', 'WE00076': 'G004', 'WE00077': 'G005', 'WE00078': 'G006', 
                        'WE00079': 'G007', 'WE00080': 'G008', 'WE00081': 'G009', 'WE00082': 'G010', 'WE00083': 'G011', 'WE00084': 'G012', 
                        'WE00085': 'H012', 'WE00086': 'H011', 'WE00087': 'H010', 'WE00088': 'H009', 'WE00089': 'H008', 'WE00090': 'H007', 
                        'WE00091': 'H006', 'WE00092': 'H005', 'WE00093': 'H004', 'WE00094': 'H003', 'WE00095': 'H002', 'WE00096': 'H001'}