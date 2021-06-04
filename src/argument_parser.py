import argparse
import os
import logging
from multiprocessing import cpu_count

LOGGER = logging.getLogger(__name__)

#TODO write extended help messages, store as string and pass to add_argument() help parameter
def parse():
    parser = argparse.ArgumentParser(description='Automated heart rate analysis of Medaka embryo videos')
    parser.add_argument('-i','--indir',     action="store",         dest='indir',       help='Input directory',                 default=False,  required = True)
    parser.add_argument('-o','--outdir',    action="store",         dest='outdir',      help='Output directory',                default=False,  required = True)
    parser.add_argument('-w','--well',      action="store",         dest='well',        help='Restrict analysis to wells',      default=None,   required = False)
    parser.add_argument('-l','--loop',      action="store",         dest='loop',        help='Restrict analysis to loop',       default=None,   required = False)
    parser.add_argument('-c','--channel',   action="store",         dest='channel',     help='Restrict analysis to channel',    default=None,   required = False)
    parser.add_argument('--cluster',        action="store_true",    dest='cluster',     help='Run analysis on a cluster',                       required = False)
    parser.add_argument('--crop',           action="store_true",    dest='crop',        help='Should crop images',                              required = False)
    parser.add_argument('--slow-mode',      action="store_true",    dest='slowmode',    help='Process all wells in slow mode',                  required = False)
    parser.add_argument('-f','--fps',       action="store",         dest='fps',         help='Frames per second',               default=0.0,    required = False, type=float)
    parser.add_argument('-p','--threads',   action="store",         dest='threads',     help='Threads to use',                  default=1,      required = False, type=int)
    parser.add_argument('-a','--average',   action="store",         dest='average',     help='average',                         default=0.0,    required = False, type=float)
    parser.set_defaults(crop=False, slowmode=False, cluster=False)
    args = parser.parse_args()

    # Adds a trailing slash if it is missing.
    args.indir  = os.path.join(args.indir, '')
    args.outdir = os.path.join(args.outdir, '')

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    return args

# Processing, done after the logger in the main file has been set up
def process(args):

    num_cores = cpu_count()
    if (num_cores > args.threads):
        LOGGER.info("the number of virtual processors in your machine is:" + str(num_cores) + "but hou have requested to run on only" + str(args.threads))
        LOGGER.info("You can have faster results (or less errors of the type \"broken pipe\") using the a ideal number of threads in the argument -p in your bash command (E.g.: -p 8). default is 1")

