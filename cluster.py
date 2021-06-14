import logging
import os
import sys

import src.io_operations    as io_operations
import src.setup            as setup

from medaka_bpm import run_algorithm


#print(sys.argv)

################################## STARTUP SETUP ##################################
# Parse input arguments
args = setup.parse_arguments()

tmp_dir = os.path.join(args.outdir, 'tmp')
well_id = 'WE00' + '{:03d}'.format(int(args.lsf_index))
analysis_id = args.channels + '-' + args.loops + '-' + well_id

setup.config_logger(tmp_dir, (analysis_id + ".log"))
arg_channels, arg_loops = setup.process_arguments(args)

LOGGER = logging.getLogger(__name__)

nr_of_videos, channels, loops = io_operations.extract_data(args.indir)

if arg_channels:
    channels    = list(arg_channels.intersection(channels))
    channels.sort()
if arg_loops:
    loops       = list(arg_loops.intersection(loops))
    loops.sort()

# Run analysis
results = {'channel': [], 'loop': [], 'well': [], 'heartbeat': []}
try:
    LOGGER.info("##### Analysis #####")
    bpm = None

    for well_frame_paths, video_metadata in io_operations.well_video_generator(args.indir, channels, loops):
        if (video_metadata['well_id'] != well_id):
            continue

        LOGGER.info("The analyse for each well can take about 10-15 minutes")
        LOGGER.info("Running....please wait...")

        try:
            bpm = run_algorithm(well_frame_paths, video_metadata, args)
            LOGGER.info("Reported BPM: " + str(bpm))

        except Exception as e:
            LOGGER.exception("Couldn't acquier BPM for well " + str(video_metadata['well_id']) 
                                + " in loop " + str(video_metadata['loop']) 
                                + " with channel " + str(video_metadata['channel']))

except Exception as e:
    LOGGER.exception("Couldn't finish analysis")
finally:
    # write bpm in tmp directory
    if bpm:
        bpm = str(bpm)
    else:
        bpm = 'NA'
    
    out_file = os.path.join(tmp_dir, (analysis_id + '.txt'))
    with open(out_file, 'w') as output:
        output.write(bpm)