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
import logging
import os
import subprocess
import shutil
import sys
import time

import glob2

import src.setup            as setup
import src.qc_statistics    as qc_statistics

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
arguments_variable = [['--' + key, str(value)] for key, value in vars(args).items() if value and value is not True]
arguments_bool = ['--' + key for key, value in vars(args).items() if value is True]
cmd_line_arguments = sum(arguments_variable, arguments_bool)

LOGGER.info("Running algorithm on datasets...")
repo_path = os.path.dirname(os.path.abspath(__file__))
main_file = os.path.join(repo_path, "medaka_bpm.py")
python_cmd = [sys.executable, main_file] + cmd_line_arguments
p = subprocess.Popen(python_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

p.wait()

# copy algorithm file
algorithm_file = os.path.join(repo_path, "src/", "segment_heart.py")
shutil.copy(algorithm_file, args.outdir)

# run statistics on algorithm
path_ground_truths = [f for f in glob2.glob(args.indir + '*.csv')][0]
indir = args.outdir
outdir = os.path.join(args.outdir, "statistics/")

# Need to split off, as in cluster mode p.wait() won't halt until the bsub job finished.
if args.cluster:
    LOGGER.info("Statistics will be run when analysis finished.")
    bsub_cmd = ['bsub', '-J', 'HR_Acc_Test', '-w', 'ended(HRConsolidated)', '-M1000', '-R', 'rusage[mem=1000]']

    if args.email == False:
        bsub_cmd += ['-o', '/dev/null']

    exe_path = os.path.join(repo_path, 'src/', 'qc_statistics.py')
    python_cmd = ['python3', exe_path, '-i', indir, '-o', outdir, '-g', path_ground_truths]

    bsub_cmd += ['source', 'activate', 'medaka_env', '&&']
    bsub_cmd += python_cmd

    LOGGER.debug(bsub_cmd)
    result = subprocess.run(bsub_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    stdout_return = result.stdout.decode('utf-8')
    LOGGER.info("\n" + stdout_return)
else:
    LOGGER.info("Peforming statistics...")
    qc_statistics.main(indir, outdir, path_ground_truths)