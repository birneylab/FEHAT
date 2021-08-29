############################################################################################################
# Authors: 
#   Sebastian Stricker, Uni Heidelberg, sebastian.stricker@stud.uni-heidelberg.de
#   Marcio Ferreira,    EMBL-EBI,       marcio@ebi.ac.uk
# Date: 08/2021
# License: Contact authors
###
# Algorithm routine that is exectued when the cluster option is specified. 
# Deviates slightly from the main file to accomodate the structure of bsub lsf dispatching.
###
############################################################################################################
import logging
import os
import sys

import src.io_operations as io_operations
import src.setup as setup

from medaka_bpm import run_algorithm

LOGGER = logging.getLogger(__name__)

################################## STARTUP SETUP ##################################
# Parse input arguments
args = setup.parse_arguments()

tmp_dir = os.path.join(args.outdir, 'tmp')

if args.lsf_index[0] == '\\':
    args.lsf_index = args.lsf_index[1:] #this is for fixing a weird behaviour: the lsf_index comes with a "\" as the first character. The "\" is usefull to pass the parameter, but need to be deleted here. 

well_id = 'WE00' + '{:03d}'.format(int(args.lsf_index))
analysis_id = args.channels + '-' + args.loops + '-' + well_id

# Check if video for well id exists before producting data.
if not io_operations.well_video_exists(args.indir, args.channels, args.loops, well_id):
    sys.exit()

setup.config_logger(tmp_dir, (analysis_id + ".log"))

# need to pass as list to generator
channels    = [args.channels]
loops       = [args.loops]

# Run analysis
results = {'channel': [], 'loop': [], 'well': [], 'heartbeat': []}
resulting_dict_from_crop = {} # NOTE: added for compatibility, not working at this moment.
try:
    LOGGER.info("##### Analysis #####")
    bpm = None

    for well_frame_paths, video_metadata in io_operations.well_video_generator(args.indir, channels, loops):
        if (video_metadata['well_id'] != well_id):
            continue

        LOGGER.info(
            "The analyse for each well can take about 1 to a few minutes")
        LOGGER.info("Running....please wait...")

        try:
            bpm = run_algorithm(well_frame_paths, video_metadata, args, resulting_dict_from_crop)
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
