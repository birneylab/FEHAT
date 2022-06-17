############################################################################################################
# Authors: 
#   Sebastian Stricker, Uni Heidelberg, sebastian.stricker@stud.uni-heidelberg.de
# Date: 08/2021
# License: Contact authors
###
# Performs an accuracy test on the current version of the algorithm.
# Run with the same arguments as you would run the normal routine.
# Indir must contain folders with experiments and a cvs file containing their ground truth values.
###
############################################################################################################
import copy
import logging
import os
import subprocess
import shutil
import sys
import time

import glob2

import qc_statistics

# Imports from base dir of repository
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import src.io_operations    as io_operations
import src.setup            as setup
from medaka_bpm             import run_multifolder

LOGGER = logging.getLogger(__name__)
args = setup.parse_arguments()

# create assessment folder & timestamped output of statistics
timestamp = time.strftime("%Y-%m-%d_%H.%M.%S")

args.outdir = os.path.join(args.outdir, "accuracy_test_" + timestamp, '')
os.makedirs(args.outdir, exist_ok=True)

# Setup Logger
setup.config_logger(args.outdir, ("assessment" + ".log"), args.debug)
LOGGER.info("Writing results into: " + args.outdir)

# run algorithm from test datasets into assessment folder
dir_list = io_operations.detect_experiment_directories(args.indir)

# Semi-automated analysis for benchmarking is done with fixed fps.
# To be as comparable as possible, the fps is set explicitly.
# The following accomodates that datasets for testing may differ in fps.
dirs_by_fps = {}
for directory in dir_list:

    #DATASET1_13FPS_171...
    #DATASET2_24FPS_171...
    name = os.path.basename(os.path.normpath(directory))
    fps = [part for part in name.split('_') if 'FPS' in part]

    assert len(fps) == 1, "FPS keyword not found. Add two digit fps info with '_##FPS_' in the directory name."
    fps = fps[0][0:2] # extract integer fps

    fps = float(fps)
    try:
        dirs_by_fps[fps].append(directory)
    except KeyError:
        dirs_by_fps[fps] = [directory]

print("Running multifolder mode. Limited console feedback, check logfiles for process status")
args_copy = copy.deepcopy(args)
for fps, dirs in dirs_by_fps.items():
    args_copy.fps = fps
    run_multifolder(args_copy, dirs)

# copy algorithm file
repo_path = parent_dir
algorithm_file = os.path.join(repo_path, "src/", "segment_heart.py")
shutil.copy(algorithm_file, args.outdir)

# run statistics on algorithm
print(args.indir)
path_ground_truths = [f for f in glob2.glob(args.indir + '*.csv')][0]
indir = args.outdir
outdir = args.outdir

if args.cluster:
    LOGGER.info("Statistics will be run when analysis finished.")
    bsub_cmd = ['bsub', '-J', 'HR_Acc_Test', '-w', 'ended(HRConsolidate-*)', '-M1000', '-R', 'rusage[mem=1000]']

    if args.email == False:
        if args.debug:
            outfile = os.path.join(outdir, 'bsub_HR_Acc_Test.log')
            bsub_cmd += ['-o', outfile]
        else:
            bsub_cmd += ['-o', '/dev/null']

    exe_path = os.path.join(repo_path, 'qc_analysis/', 'qc_statistics.py')
    python_cmd = ['python3', exe_path, '-i', indir, '-o', outdir, '-g', path_ground_truths]

    bsub_cmd += python_cmd

    LOGGER.debug(bsub_cmd)
    result = subprocess.run(bsub_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    stdout_return = result.stdout.decode('utf-8')
    LOGGER.info("\n" + stdout_return)
else:
    LOGGER.info("Peforming statistics...")
    qc_statistics.main(indir, outdir, path_ground_truths)