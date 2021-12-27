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

from numpy import isnan as isnan
import pandas as pd

from medaka_bpm import analyse

LOGGER = logging.getLogger(__name__)

################################## STARTUP SETUP ##################################
try:
    # Parse input arguments
    args = setup.parse_arguments()

    experiment_id, args = setup.process_arguments(args, is_cluster_node=True)

    # Should receive only a single channel/loop/wellID, it's a cluster node with 1 job.
    # Needed as list though
    channels = list(args.channels)
    loops = list(args.loops)

    tmp_dir = os.path.join(args.outdir, 'tmp')

    if args.lsf_index[0] == '\\':
        # this is for fixing a weird behaviour: the lsf_index comes with a "\" as the first character. The "\" is usefull to pass the parameter, but need to be deleted here.
        args.lsf_index = args.lsf_index[1:]

    well_id = 'WE00' + '{:03d}'.format(int(args.lsf_index))
    analysis_id = channels[0] + '-' + loops[0] + '-' + well_id

    # Check if video for well id exists before producting data.
    if not io_operations.well_video_exists(args.indir, channels[0], loops[0], well_id):
        sys.exit()

    setup.config_logger(tmp_dir, (analysis_id + ".log"), args.debug)

    # Run analysisp
    well_nr = int(well_id[-2:])
    results = analyse(args, channels, loops, wells=[well_nr])

    # write bpm in tmp directory
    out_string = ""
    for col in results.columns:
        value = results[col][0]

        if not pd.isnull(value):
            value = str(value)
        else:
            value = 'NA'

        out_string += f"{col}:{value};"

    out_string = out_string[:-1]

    out_file = os.path.join(tmp_dir, (analysis_id + '.txt'))
    with open(out_file, 'w') as output:
        output.write(out_string)

except Exception as e:
    LOGGER.exception("In analysis dispatched to a cluster node")