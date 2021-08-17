import logging
import os

import csv

import glob2
import cv2

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
    subdir_list = {os.path.join(os.path.dirname(p), '') for p in glob2.glob(indir + '/*/')} # set([os.path.dirname(p) for p in glob2.glob(indir + '/*/')])
    
    # Condition: Tiffs inside or croppedRAWTifffolder
    for path in subdir_list:
        if os.path.basename(os.path.normpath(path)) == 'croppedRAWTiff':
            continue

        cond_1 = os.path.isdir(os.path.join(path, "croppedRAWTiff"))
        cond_2 = glob2.glob(path + '*.tif') + glob2.glob(path + '*.tiff')
        if cond_1 or cond_2:
            subdirs.add(path)

    # No subdirectories. Add indir as only folder
    if not subdirs:
        subdirs.add(indir)
        
    return subdirs

# TODO: get default to greyscale, as the videos are only in greyscale, conversion everywhere is overhead
def load_well_video_8bits(frame_paths_sorted, max_frames = -1):
    LOGGER.info("Loading video as 8 bits")
    video8 = []
    for index, path in enumerate(frame_paths_sorted):
        #check if the function is supposed to read every frame (-1), otherwise, runs only the number of frames specified in max_frames argument
        # it is usefull as if we need crop and save the images, we only need the first 5 frames to make the average of position.
        if max_frames == -1 or index < max_frames:
            frame = cv2.imread(path, 0) # 0 flag to read image as bw
            video8.append(frame)
    return video8

def load_well_video_16bits(frame_paths_sorted):
    LOGGER.info("Loading video as 16 bits")
    video16 = []
    for path in frame_paths_sorted:       
        frame = cv2.imread(path, -1) # -1 flag to read image as it is (16 bits)
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
    tiffs = glob2.glob(indir + '*SL001' + '*.tif') + glob2.glob(indir + '*SL001' + '*.tiff')
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
#TODO: Transfer functionality into pandas dataframes. Probably more stable and clearer
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
