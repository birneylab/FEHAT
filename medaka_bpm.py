#!/usr/bin/env python3
############################################################################################################
### Author: Sebastian Stricker, Uni Heidelberg, sebastian.stricker@stud.uni-heidelberg.de
### Date: 04/2021
### License: Contact author
###
### Main program file.
### 
############################################################################################################
import logging
import os
import time
import src.io_operations    as io_operations
import src.argument_parser  as argument_parser
import src.segment_heart    as segment_heart

################################## INPUT ##################################

# indir = "/run/media/nase/Local Disk AGW/test_set_v2/"
# outdir = os.path.join(indir, "outdir")
# fps = 13.0
# crop = True

################################## SETUP ##################################
# Parse input arguments
args = argument_parser.parse()

# Global logger settings
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler(os.path.join(args.outdir,"medaka_bpm_script.log"))
                    ])

LOGGER = logging.getLogger(__name__)

argument_parser.process(args)

total_time = time.time()

################################## MAIN PROGRAM START ##################################
LOGGER.info("##### MedakaBPM #####")

nr_of_videos, channels, loops = io_operations.extract_data(args.indir)

# Extract Video metadata
LOGGER.info("Deduced number of videos: " + str(nr_of_videos))
LOGGER.info("Deduced Channels: " + ', '.join(channels))
LOGGER.info("Deduced number of Loops: " + str(len(loops)) + "\n")

################################## ANALYSIS ##################################
results = {'channel': [], 'loop': [], 'well': [], 'heartbeat': []}
analysis_times = []
try:
    LOGGER.info("##### Analysis #####")
    # TODO: split program flow here for cluster and single machine execution
    for well_frame_paths, video_metadata in io_operations.well_video_generator(args.indir, channels, loops):
        LOGGER.info("The analyse for each well can take about 10-15 minutes\n")
        LOGGER.info("Running....please wait...")
        try:
            # TODO: I/O is very slow, 1 video ~500mb ~20s locally. Buffer video loading?
            # Load video
            LOGGER.info("Loading video")
            video = io_operations.load_well_video(well_frame_paths)
            video_metadata['timestamps'] = io_operations.extract_timestamps(well_frame_paths)

            bpm = None

            # Crop and analyse
            if args.crop:
                video = segment_heart.crop(video)
            bpm = segment_heart.run(video, vars(args), video_metadata)

            LOGGER.info("Reported BPM: " + str(bpm))

        except Exception as e:
            LOGGER.exception("Couldn't acquier BPM for well " + str(video_metadata['well_id']) 
                                + " in loop " + str(video_metadata['loop']) 
                                + " with channel " + str(video_metadata['channel']))
        finally:
            # Save results
            results['channel'].append(video_metadata['channel'])
            results['loop'].append(video_metadata['loop'])
            results['well'].append(video_metadata['well_id'])
            results['heartbeat'].append(bpm)

except Exception as e:
    LOGGER.exception("Couldn't finish analysis")

################################## OUTPUT ##################################
LOGGER.info("#######################")
LOGGER.info("Finished analysis")
nr_of_results = len(results['heartbeat'])
if (nr_of_videos is not nr_of_results):
    LOGGER.warning("Logic fault. Number of results (" + str(nr_of_results) + ") doesn't match number of videos detected (" + str(nr_of_videos) + ")")

io_operations.write_to_spreadsheet(args.outdir, results)