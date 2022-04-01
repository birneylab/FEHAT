############################################################################################################
### Author: Sebastian Stricker, Uni Heidelberg, sebastian.stricker@stud.uni-heidelberg.de
###         Marcio Ferreira,    EMBL-EBI,       marcio@ebi.ac.uk
### Date: 08/2021
### License: Contact author
###
### Configures logger and handles command line arguments.
### 
############################################################################################################
import argparse
import os
import logging

import configparser
# Read config
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_dir, 'config.ini')

config = configparser.ConfigParser()
config.read(config_path)

LOGGER = logging.getLogger(__name__)

def config_logger(logfile_path, logfile_name="medaka_outdir.log", in_debug_mode=False):
    os.makedirs(logfile_path, exist_ok=True)

    loglevel = logging.INFO
    if in_debug_mode:
        loglevel = logging.DEBUG

    # Global logger settings
    logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=loglevel,
                        handlers=[
                            logging.StreamHandler(),
                            logging.FileHandler(os.path.join(
                                logfile_path, logfile_name))
                        ])

# TODO write extended help messages, store as string and pass to add_argument() help parameter
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Automated heart rate analysis of Medaka embryo videos')
    # General analysis arguments
    parser.add_argument('-i', '--indir',     action="store",         dest='indir',
                        help='Input directory',                                 default=False,      required=True)
    parser.add_argument('-o', '--outdir',    action="store",         dest='outdir',
                        help='Output directory. Default=indir',                 default=False,      required=False)
    parser.add_argument('-w', '--wells',     action="store",         dest='wells',
                        help='Restrict analysis to wells',                      default='[1-96]',   required=False)
    parser.add_argument('-l', '--loops',     action="store",         dest='loops',
                        help='Restrict analysis to loop',                       default=None,       required=False)
    parser.add_argument('-c', '--channels',  action="store",         dest='channels',
                        help='Restrict analysis to channel',                    default=None,       required=False)
    parser.add_argument('-f', '--fps',       action="store",         dest='fps',
                        help='Frames per second',                               default=0.0,        required=False,   type=float)

    # Cropping Arguments
    parser.add_argument('-s', '--embryo_size', action="store",         dest='embryo_size',
                        help='radius of embryo in pixels',                  default=300,        required=False, type=int)
    parser.add_argument('--crop',           action="store_true",    dest='crop',
                        help='Should crop images and analyse',                                  required=False)
    parser.add_argument('--crop_and_save',  action="store_true",    dest='crop_and_save',
                        help='Should crop crop images and save, and run bpm in cropped images', required=False)
    parser.add_argument('--only_crop',      action="store_true",    dest='only_crop',
                        help='Should only crop images, not run bpm script',                     required=False)

    # Cluster arguments. Index is hidden argument that is set through bash script to assign wells to cluster instances.
    parser.add_argument('--cluster',        action="store_true",    dest='cluster',
                        help='Run analysis on a cluster',                       required=False)
    parser.add_argument('--email',          action="store_true",    dest='email',
                        help='Receive email for cluster notification',          required=False)
    parser.add_argument('-m', '--maxjobs',   action="store",         dest='maxjobs',
                        help='maxjobs on the cluster',          default=None,   required=False)
    parser.add_argument('-x', '--lsf_index', action="store",         dest='lsf_index',
                        help=argparse.SUPPRESS,                                 required=False)

    # Debug flag
    parser.add_argument('--debug',          action="store_true",    dest='debug',
                        help='Additional debug output',                          required=False)
    parser.set_defaults(crop=False, only_crop=False, crop_and_save=False,
                        slowmode=False, cluster=False, email=False, debug=False)
    args = parser.parse_args()

    # Adds a trailing slash if it is missing.
    args.indir = os.path.join(args.indir, '')
    if args.outdir:
        args.outdir = os.path.join(args.outdir, '')

    return args

# Processing, done after the logger in the main file has been set up
def process_arguments(args, is_cluster_node=False):
    will_crop = (args.only_crop or args.crop_and_save or args.crop)

    # Move up one folder if croppedRAWTiff was given. Experiment folder is above it.
    if os.path.basename(os.path.normpath(args.indir)) == "croppedRAWTiff":
        args.indir = os.path.dirname(os.path.normpath(args.indir))

    # Output into experiment folder, if no ouput was given
    if not args.outdir:
        args.outdir = args.indir

    # Get the experiment folder name.
    experiment_name = os.path.basename(os.path.normpath(args.indir))

    # experiment_id: Number code for logfile and outfile respectively
    # e.g.: 170814162619_Ol_SCN5A_NKX2_5_Temp_35C -> 170814162619
    experiment_id = experiment_name.split('_')[0]

    # Outdir should be named after experiment.
    # Do not do for cluster nodes, already created on dispatch
    if not is_cluster_node:
        
        software_version = config['DEFAULT']['VERSION']

        # Outdir should start with experiment name
        args.outdir = os.path.join(args.outdir, f"{experiment_name}_medaka_bpm_out_{software_version}", '')
        os.makedirs(args.outdir, exist_ok=True)

    # croppedRAWTiff folder present? Use cropped files for analysis
    if os.path.isdir(os.path.join(args.indir, "croppedRAWTiff")):
        args.indir = os.path.join(args.indir, "croppedRAWTiff", '')

    if not args.maxjobs:
        args.maxjobs = ''
    else:
        args.maxjobs = '%' + args.maxjobs

    if args.channels:
        args.channels = {c for c in args.channels.split('.')}
    if args.loops:
        args.loops = {l for l in args.loops.split('.')}

    return experiment_id, args
