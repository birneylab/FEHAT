#!/usr/bin/env python3
############################################################################################################
### Author: Sebastian Stricker, Uni Heidelberg, sebastian.stricker@stud.uni-heidelberg.de
### Date: 04/2021
### License: Contact author
###
### Main program file.
### 
############################################################################################################
import gc

import logging
import os
import subprocess
import sys

import src.io_operations    as io_operations
import src.setup            as setup
import src.segment_heart    as segment_heart

LOGGER = logging.getLogger(__name__)

################################## ALGORITHM ##################################
def run_algorithm(well_frame_paths, video_metadata, args):
    LOGGER.info("Analysing video - "    + "Channel: "   + str(video_metadata['channel'])
                                        + " Loop: "      + str(video_metadata['loop'])
                                        + " Well: "      + str(video_metadata['well_id'])
    )

    # TODO: I/O is very slow, 1 video ~500mb ~20s locally. Buffer video loading for single machine?
    # Load video
    video = io_operations.load_well_video(well_frame_paths)
    video_metadata['timestamps'] = io_operations.extract_timestamps(well_frame_paths)

    # Crop and analyse
    if args.crop:
        video = segment_heart.crop(video)
    bpm = segment_heart.run(video, vars(args), video_metadata)

    return bpm;

def run_multifolder(args, dirs):
    # Case 1: No cropped tiff, no other folders, tiffs contained contained
    # If it's croppedTiff folder
    # -> main(args)

    # # Case 2: Multiple experiment folders
    # print("Calling multiple folders")
    # #loop over directories
    #     # Dictionary of the aruments
    #     arguments_variable = {['--' + key, str(value)] for key, value in vars(args).items() if value and value is not True}
    #     arguments_bool = ['--' + key for key, value in vars(args).items() if value is True]
        
    #     # Change the indir-argument to the subfolder
    #     # Change the outdir to outdir + subfolder_name
    #     # If it's croppedTiff folder

    #     arguments = sum(arguments_variable, arguments_bool)
    #     main(args)
    #     args = list(args)
    #     python_cmd = ['python3', 'medake_bpm.py', arguments]
    #     subprocess.run(python_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # processes to be dispatched
    cmd_list = []
    procs_list = []

    # loop throw the folders
    for path in dir_list:   
    
        # get the indir and outdir arguments on the fly
        args.indir = path

        # get arguments for recursive call
        arguments_variable = [['--' + key, str(value)] for key, value in vars(args).items() if value and value is not True]
        arguments_bool = ['--' + key for key, value in vars(args).items() if value is True]
        arguments = sum(arguments_variable, arguments_bool)

        # absolute filepath and sys.executeable for windows compatibility
        filename = os.path.abspath(__file__)
        python_cmd = [sys.executable, filename] + arguments
        cmd_list.append(python_cmd)
    
    if args.cluster:
        for cmd in cmd_list:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        max_subprocesses = 2
        print("Processing subfolders " + str(max_subprocesses) + " at a time.")
        i = max_subprocesses
        for cmd in cmd_list:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            procs_list.append(p)

            experiment_name = cmd[cmd.index("--indir")+1]
            experiment_name = os.path.basename(os.path.normpath(experiment_name))
            print("Starting " + experiment_name)
            i -= 1

            if i == 0:
                for proc in procs_list:
                    proc.wait()
                print("Finished process set\n")
                i = max_subprocesses
                
        for proc in procs_list:
            proc.wait()
        print("\nFinished all subfolders")

def main(args):
    ################################## STARTUP SETUP ##################################
    arg_channels, arg_loops, experiment_id = setup.process_arguments(args)
    setup.config_logger(args.outdir, ("logfile_" + experiment_id + ".log"))

    ################################## MAIN PROGRAM START ##################################
    LOGGER.info("##### MedakaBPM #####")

    nr_of_videos, channels, loops = io_operations.extract_data(args.indir)
    if arg_channels:
        channels    = list(arg_channels.intersection(channels))
        channels.sort() 
    if arg_loops:
        loops       = list(arg_loops.intersection(loops))
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
        #Run cluster analysis
        LOGGER.info("Running on cluster")
        try:
            for channel in channels:
                for loop in loops:
                    LOGGER.info("Dispatching wells from " + channel + " " + loop + " to cluster")

                    # Prepae arguments to pass to bsub job
                    args.channels   = channel
                    args.loops      = loop

                    arguments_variable = [['--' + key, str(value)] for key, value in vars(args).items() if value and value is not True]
                    arguments_bool = ['--' + key for key, value in vars(args).items() if value is True]
                    arguments = sum(arguments_variable, arguments_bool)

                    # pass arguments down. Add Jobindex to assign cluster instances to specific wells.
                    python_cmd = ['python3', 'cluster.py'] + arguments + ['-x', '\$LSB_JOBINDEX']

                    jobname = 'heartRate' + args.wells + str(args.maxjobs)

                    bsub_cmd = ['bsub', '-J', jobname, '-M20000', '-R', 'rusage[mem=8000]']                   

                    if args.email == False:                        
                        bsub_cmd.append( '-o /dev/null')                  

                    cmd = bsub_cmd + python_cmd

                    #Create a job array for each well
                    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

                    LOGGER.info("\n"+ result.stdout.decode('utf-8'))

            #Create a dependent job for final report
            # #bsub -J "consolidateHeartRate" -w "ended(heartRate)"  -M3000 -R rusage[mem=3000] $email python3 consolidated.py -i "$out_dir" -o "$out_dir" #-o log_consolidated.txt

            #error here
            consolidate_cmd = ['bsub', '-J', 'HRConsolidated', '-w', 'ended(heartRate)', '-M3000', '-R', 'rusage[mem=3000]'] # changed the job name so it can be seen in list of jobs

            if args.email == False:
                consolidate_cmd.append('-o /dev/null')

            tmp_dir = os.path.join(args.outdir, 'tmp')
            python_cmd = ['python3', 'src/cluster_consolidate.py', '-i', tmp_dir, '-o', args.outdir]

            consolidate_cmd += python_cmd            

            subprocess.run(consolidate_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        except Exception as e:
            LOGGER.exception("During dispatching of jobs onto the cluster")

    else:
        LOGGER.info("Running on a single machine")
        results = {'channel': [], 'loop': [], 'well': [], 'heartbeat': []}
        try:
            LOGGER.info("##### Analysis #####")

            for well_frame_paths, video_metadata in io_operations.well_video_generator(args.indir, channels, loops):
                LOGGER.info("The analyse for each well can take about from one to several minutes\n")
                LOGGER.info("Running....please wait...")

                bpm = None

                try:
                    bpm = run_algorithm(well_frame_paths, video_metadata, args)
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

                    gc.collect()

        except Exception as e:
            LOGGER.exception("Couldn't finish analysis")

        ################################## OUTPUT ##################################
        LOGGER.info("#######################")
        LOGGER.info("Finished analysis")
        nr_of_results = len(results['heartbeat'])
        if (nr_of_videos != nr_of_results):
            LOGGER.warning("Logic fault. Number of results (" + str(nr_of_results) + ") doesn't match number of videos detected (" + str(nr_of_videos) + ")")

        io_operations.write_to_spreadsheet(args.outdir, results, experiment_id)

# TODO: Workaround to import run_algorithm into cluster.py. Maybe solve more elegantly
if __name__ == '__main__':
    # Parse input arguments.
    args = setup.parse_arguments()

    ################# MULTI FOLDER DETECTION ######################
    # Detect subfolders in indir
    dir_list = io_operations.detect_experiment_directories(args.indir)

    if len(dir_list) < 1:
        print("Error: Indirectory invalid")
        sys.exit(1)

    if len(dir_list) == 1:          # only one directory: process directly
        args.indir = dir_list.pop()
        main(args)
    else:                            # Multidir. Process separately
        print("Running multifolder mode. Limited console feedback, check logfiles for process status")
        run_multifolder(args, dir_list)