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

            #Well
            for well_id in range(1,97):
                well_id = ('WE00' + '{:03d}'.format(well_id))

                well_frames = [frame for frame in loop_videos if well_id in frame]

                # Skipping costs (close to) no time and the logic is simpler than extracting the exact set of wells per loop/channel.
                # Could be solved cleaner though.
                if not well_frames:
                    continue
                
                # Sort frames in correct order
                frame_indices = [frameIdx(path) for path in well_frames]
                _, well_frames_sorted = (list(t) for t in zip(*sorted(zip(frame_indices, well_frames))))

                metadata = {'well_id': well_id, 'loop': loop, 'channel': channel}
                yield well_frames_sorted, metadata

# TODO: get default to greyscale, as the videos are only in greyscale, conversion everywhere is overhead
def load_well_video(frame_paths_sorted, color_mode = cv2.IMREAD_COLOR):
    LOGGER.info("Loading video")
    video = []
    for path in frame_paths_sorted:
        frame = cv2.imread(path, flags=color_mode)
        video.append(frame)

    return video

def extract_timestamps(sorted_frame_paths):
    # splits every path at '-T'. Picks first 10 chars of the string that starts with a number.
    timestamps = [[s for s in path.split('-T') if s[0].isdigit()][-1][0:10] for path in sorted_frame_paths]
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
    channels = {'CO' + tiff.split('-CO')[-1][0] for tiff in tiffs} # using a set, gives only unique values
    channels = sorted(list(channels))

    # Extract different Loops
    loops = {'LO' + tiff.split('-LO')[-1][0:3] for tiff in tiffs} # using a set, gives only unique values
    loops = sorted(list(loops))

    return nr_of_videos, channels, loops

# From Tim-script
def frameIdx(path):
    idx = path.split('-SL')[-1]
    idx = idx.split('-')[0]
    idx = int(idx)
    return idx

# Results:
#   Dictionary {'channel': [], 'loop': [], 'well': [], 'heartbeat': []}
#TODO: Transfer functionality into pandas dataframes. Probably more stable and clearer
def write_to_spreadsheet(outdir, results):
    LOGGER.info("Saving acquired data to spreadsheet")
    outpath = os.path.join(outdir, "results.csv")

    # Don't erase previous results by accident
    if os.path.isfile(outpath):
        LOGGER.warning("Outdir already contains results file. Writing new file version")

    version = 2
    while os.path.isfile(outpath):
        outpath = os.path.join(outdir, "results_v" + str(version) + ".csv")
        version += 1

    # Write results in file
    with open(outpath, 'w') as outfile:
        writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        writer.writerow(['Index', 'WellID', 'Loop', 'Channel','Heartrate (BPM)'])
        nr_of_results = len(results['heartbeat'])
        for idx in range(nr_of_results):
                ch      = results['channel'][idx]
                loop    = results['loop'][idx]
                well    = results['well'][idx]
                bpm     = results['heartbeat'][idx]
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