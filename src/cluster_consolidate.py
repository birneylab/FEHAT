#!/usr/bin/env python
############################################################################################################
# Authors: 
#   Marcio Ferreira,    EMBL-EBI,       marcio@ebi.ac.uk
#   Sebastian Stricker, Uni Heidelberg, sebastian.stricker@stud.uni-heidelberg.de
# Date: 08/2021
# License: Contact authors
###
# For cluster mode.
# Each node creates it's own results file in outdir/tmp.
# Called as a dependend bsub job when all instances running cluster.py have finished.
# Gathers data from outdir/tmp and creates the final report.
###
############################################################################################################
import argparse
import pandas as pd
from pathlib import Path

from pathlib import Path
import logging

import io_operations
import setup

LOGGER = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Read in medaka heart video frames')
parser.add_argument('-o','--outdir', action="store", dest='outdir', help='Where to store the output report and the global log',    default=False, required = True)
parser.add_argument('-i','--indir', action="store", dest='indir', help='Path to temp folder with results',                      default=False, required = True)

args = parser.parse_args()

args.indir  = Path(args.indir)
args.outdir = Path(args.outdir)

# Number code for logfile and outfile respectively
experiment_id = args.outdir.name.split('_')[0]

setup.config_logger(args.outdir, ("logfile_" + experiment_id + ".log"))

try:
    LOGGER.info("Consolidating cluster results")

    # path to central log file
    logs_paths    = args.indir.glob('*.log')
    results_paths = args.indir.glob('*.txt')

    # log files and results files of one analysis should have same index in lists
    logs_paths.sort()
    results_paths.sort()

    results = pd.DataFrame()
    
    # Extract data from all files in tmp dir into dataframe
    for log, result in zip(logs_paths, results_paths):
        well_result = {}
        if (Path(log).stem.split('-') != Path(result).stem.split('-')):
            LOGGER.exception("Logfile and result file order not right")

        with open(log) as fp:
            log_text = fp.read()
            well_result['log'] = log_text
        
        # File stores column name and value in the following format
        # (cluster.py) out_string = "heartbeat:123;Heart Size:1110;HROI count:2; ..."
        with open(result) as fp:
            out_string = fp.read()
            fields = out_string.split(';')
            for field in fields:
                entry = field.split(':')
                well_result[entry[0]] = entry[1]

        results = results.append(well_result, ignore_index=True)

    # Sort entries for output
    results = pd.DataFrame.from_dict(results)
    results = results.sort_values(by=['channel', 'loop', 'well_id'])

    # Consolidate all logs into the general log file
    logs = results['log'].tolist()
    LOGGER.info("Log reports from analyses: \n" + '\n'.join(logs))

    # Logs should not appear in the output csv
    results = results.drop(columns=['log'])
    io_operations.write_to_spreadsheet(args.outdir, results, experiment_id)

except Exception as e:
    LOGGER.exception("Couldn't consolidate results from cluster analysis")