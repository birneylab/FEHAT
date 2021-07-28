#!/usr/bin/env python3
############################################################################################################
# Author: Sebastian Stricker, Uni Heidelberg, sebastian.stricker@stud.uni-heidelberg.de
# Date: 04/2021
# License: Contact author
###
# Main program file.
###
############################################################################################################
import gc

import logging
import os
import subprocess
import sys

import src.io_operations as io_operations
import src.setup as setup
import src.segment_heart as segment_heart

LOGGER = logging.getLogger(__name__)

################################## ALGORITHM ##################################


def run_algorithm(well_frame_paths, video_metadata, args):
    LOGGER.info("Analysing video - " + "Channel: " + str(video_metadata['channel'])
                + " Loop: " + str(video_metadata['loop'])
                + " Well: " +
                str(video_metadata['well_id'])
                )

    # TODO: I/O is very slow, 1 video ~500mb ~20s locally. Buffer video loading for single machine?
    # Load video
    video = io_operations.load_well_video(well_frame_paths)
    video_metadata['timestamps'] = io_operations.extract_timestamps(
        well_frame_paths)

    # Crop and analyse
    if args.crop:
        video = segment_heart.crop(video)
    bpm = segment_heart.run(video, vars(args), video_metadata)

    return bpm


def main(args):

    ################################## STARTUP SETUP ##################################
    setup.config_logger(args.outdir)
    arg_channels, arg_loops = setup.process_arguments(args)

    ################# MULTI FOLDER DETECTION ######################

    # Try to detect subfolder in indir
    subdir_list = set([os.path.dirname(p)
                      for p in glob.glob(args.indir + '/*/*')])

    if len(subdir_list) > 1:
        # There are more than one folder in indir
        LOGGER.info("There are " + str(len(subdir_list)) +
                    " folders in your indir. Trying to read each one as a separated experiment...")

    # just store the original outdir parh
    temp_outdir = vars(args)['outdir']

    for path in subdir_list:   # loop throw the folders

        # get the indir and outdir arguments on the fly
        vars(args)['indir'] = path

        # and concatenate the specific subfolder name to the outdir momentarily
        vars(args)['outdir'] = temp_outdir + '/' + \
            os.path.basename(os.path.normpath(vars(args)['indir']))
        temp_indir = vars(args)['indir']

        # verify if tiff files are still in another subfolder, as fopr example, "CroppedTiff" or whatever
        for fname in os.listdir(vars(args)['indir']):
            # If not, do nothing  and goes to the run script
            if fname.endswith('.tif') or fname.endswith('.tiff'):
                break
            else:
                # if yes, access this folder and look for tiff images
                vars(args)['indir'] = temp_indir + '/' + \
                    str(next(os.walk(vars(args)['indir']))[1][0]) + '/'

    ################################## MAIN PROGRAM START ##################################
    LOGGER.info("##### MedakaBPM #####")

    nr_of_videos, channels, loops = io_operations.extract_data(args.indir)
    if arg_channels:
        channels = list(arg_channels.intersection(channels))
        channels.sort()
    if arg_loops:
        loops = list(arg_loops.intersection(loops))
        loops.sort()

    if not loops or not channels:
        LOGGER.error("No loops or channels were found!")
        sys.exit()

    # Extract Video metadata
    LOGGER.info("Deduced number of videos: " + str(nr_of_videos))
    LOGGER.info("Deduced Channels: " + ', '.join(channels))
    LOGGER.info("Deduced number of Loops: " + str(len(loops)) + "\n")

    ################################## ANALYSIS ##################################
    if args.cluster:
        # Run cluster analysis
        LOGGER.info("Running on cluster")
        try:
            for channel in channels:
                for loop in loops:
                    LOGGER.info("Dispatching wells from " +
                                channel + " " + loop + " to cluster")

                    # Prepae arguments to pass to bsub job
                    args.channels = channel
                    args.loops = loop

                    arguments_variable = [
                        ['--' + key, str(value)] for key, value in vars(args).items() if value and value is not True]
                    arguments_bool = ['--' + key for key,
                                      value in vars(args).items() if value is True]
                    arguments = sum(arguments_variable, arguments_bool)

                    # pass arguments down. Add Jobindex to assign cluster instances to specific wells.
                    python_cmd = ['python3', 'cluster.py'] + \
                        arguments + ['-x', '\$LSB_JOBINDEX']

                    jobname = 'heartRate' + args.wells + str(args.maxjobs)

                    bsub_cmd = ['bsub', '-J', jobname,
                                '-M20000', '-R', 'rusage[mem=8000]']

                    if args.email == False:
                        bsub_cmd.append('-o /dev/null')

                    cmd = bsub_cmd + python_cmd

                    # Create a job array for each well
                    result = subprocess.run(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

                    LOGGER.info("\n" + result.stdout.decode('utf-8'))

            # Create a dependent job for final report
            # #bsub -J "consolidateHeartRate" -w "ended(heartRate)"  -M3000 -R rusage[mem=3000] $email python3 consolidated.py -i "$out_dir" -o "$out_dir" #-o log_consolidated.txt

            # error here
            # changed the job name so it can be seen in list of jobs
            consolidate_cmd = ['bsub', '-J', 'HRConsolidated', '-w',
                               'ended(heartRate)', '-M3000', '-R', 'rusage[mem=3000]']

            if args.email == False:
                consolidate_cmd.append('-o /dev/null')

            tmp_dir = os.path.join(args.outdir, 'tmp')
            python_cmd = ['python3', 'src/cluster_consolidate.py',
                          '-i', tmp_dir, '-o', args.outdir]

            consolidate_cmd += python_cmd

            subprocess.run(consolidate_cmd, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)

        except Exception as e:
            LOGGER.exception("During dispatching of jobs onto the cluster")

    else:
        LOGGER.info("Running on a single machine")
        results = {'channel': [], 'loop': [], 'well': [], 'heartbeat': []}
        try:
            LOGGER.info("##### Analysis #####")

            for well_frame_paths, video_metadata in io_operations.well_video_generator(args.indir, channels, loops):
                LOGGER.info(
                    "The analyse for each well can take about from one to several minutes\n")
                LOGGER.info("Running....please wait...")

                bpm = None

                try:
                    bpm = run_algorithm(well_frame_paths, video_metadata, args)
                    LOGGER.info("Reported BPM: " + str(bpm))

                except Exception as e:
                    LOGGER.exception("Couldn't acquier BPM for well " + str(video_metadata['well_id'])
                                     + " in loop " +
                                     str(video_metadata['loop'])
                                     + " with channel " + str(video_metadata['channel']))
                finally:
                    # Save results
                    results['channel'].append(video_metadata['channel'])
                    results['loop'].append(video_metadata['loop'])
                    results['well'].append(video_metadata['well_id'])
                    results['heartbeat'].append(bpm)

                    gc.collect()

        except Exception as e:
            LOGGER.exception("Couldn't finish analysis")

        ################################## OUTPUT ##################################
        LOGGER.info("#######################")
        LOGGER.info("Finished analysis")
        nr_of_results = len(results['heartbeat'])
        if (nr_of_videos != nr_of_results):
            LOGGER.warning("Logic fault. Number of results (" + str(nr_of_results) +
                           ") doesn't match number of videos detected (" + str(nr_of_videos) + ")")

        io_operations.write_to_spreadsheet(args.outdir, results)


# TODO: Workaround to import run_algorithm into cluster.py. Maybe solve more elegantly
if __name__ == '__main__':
    # Parse input arguments.
    args = setup.parse_arguments()

    # TODO: Handle different directory structures here.
    # Should be simple to change the indir and outdir in 'args' and pass for each detected directory respectively.
    # Avoid cluttering this file and write functions to detect and return all input/output directories in /src/io_operations.py
    main(args)
