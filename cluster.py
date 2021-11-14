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
try:
    # Parse input arguments
    args = setup.parse_arguments()

    tmp_dir = os.path.join(args.outdir, 'tmp')

    if args.lsf_index[0] == '\\':
        # this is for fixing a weird behaviour: the lsf_index comes with a "\" as the first character. The "\" is usefull to pass the parameter, but need to be deleted here.
        args.lsf_index = args.lsf_index[1:]

    well_id = 'WE00' + '{:03d}'.format(int(args.lsf_index))
    analysis_id = args.channels + '-' + args.loops + '-' + well_id

    # check if there is a croppedRawTiff folder  ( I don´t know how to solve more elegantly, as the 'args = setup.parse_arguments()' above get the indir originally passed as argument, even if we change it later)
    subdir_list = os.listdir(args.indir)
    for n in subdir_list:
        if 'croppedRAWTiff' in n:
            args.indir = os.path.join(args.indir, 'croppedRAWTiff/', '')

    # Check if video for well id exists before producting data. Also check in CroppedRAWTiff folder.
    if not io_operations.well_video_exists(args.indir, args.channels, args.loops, well_id):
        sys.exit()

    setup.config_logger(tmp_dir, (analysis_id + ".log"), args.debug)

    # need to pass as list to generator
    channels = [args.channels]
    loops = [args.loops]

    # Run analysis
    results = { 'channel':          [], 
                'loop':             [],
                'well':             [], 
                'heartbeat':        [],
                'Heart size':       [], # qc_attributes
                'HROI count':       [],
                'Stop frame':       [],
                'Number of peaks':  [],
                'Prominence':       [],
                'Height':           [],
                'Low variance':     []}

    # NOTE: added for compatibility, not working at this moment.
    resulting_dict_from_crop = {}
    try:
        LOGGER.info("##### Analysis #####")
        bpm = None
        fps = None
        qc_attributes = {   "Heart size": None, 
                            "HROI count": None, 
                            "Stop frame": None, 
                            "Number of peaks": None,
                            "Prominence": None,
                            "Height": None,
                            "Low variance": None}

        for well_frame_paths, video_metadata in io_operations.well_video_generator(args.indir, channels, loops):
            if (video_metadata['well_id'] != well_id):
                continue

            LOGGER.info(
                "The analyse for each well can take about 1 to a few minutes")
            LOGGER.info("Running....please wait...")

            try:
                bpm, fps, qc_attributes = run_algorithm(well_frame_paths, video_metadata, args, resulting_dict_from_crop)
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

        if fps:
            fps = str(fps)
        else:
            fps = 'NA'

        out_string = "heartbeat:" + bpm + ";fps:" + fps

        # Output quality control attributes only in debug mode
        if args.debug:
            for key, value in qc_attributes.items():
                out_string += (";" + str(key) + ':' + str(value))
        else:
            out_string += ";"

        out_file = os.path.join(tmp_dir, (analysis_id + '.txt'))
        with open(out_file, 'w') as output:
            output.write(out_string)
except Exception as e:
    LOGGER.exception("In analysis dispatched to a cluster node")