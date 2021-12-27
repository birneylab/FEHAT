#!/usr/bin/env python3
############################################################################################################
# Authors:
#   Sebastian Stricker, Uni Heidelberg, sebastian.stricker@stud.uni-heidelberg.de
#   Marcio Ferreira,    EMBL-EBI,       marcio@ebi.ac.uk
# Date: 08/2021
# License: Contact authors
###
# Main program file.
###
############################################################################################################
import gc

import logging
import os
import subprocess
import sys

import pandas as pd

import src.io_operations as io_operations
import src.setup as setup
import src.segment_heart as segment_heart

SOFTWARE_VERSION = "1.2.1 (dec21)"

LOGGER = logging.getLogger(__name__)


################################## ALGORITHM ##################################

# Analyse a range of wells
def analyse(args, channels, loops):
    LOGGER.info("##### Analysis #####")
    # Results for all wells
    results = pd.DataFrame()

    try:
        resulting_dict_from_crop = {}
        for well_frame_paths, video_metadata in io_operations.well_video_generator(args.indir, channels, loops):
            LOGGER.info("The analysis for each well can take one to several minutes\n")
            LOGGER.info("Running....please wait...")

            # Results of current well
            well_result = {}
            bpm = None
            fps = None
            qc_attributes = {}
            
            try:
                bpm, fps, qc_attributes = run_algorithm(well_frame_paths, video_metadata, args, resulting_dict_from_crop)
                LOGGER.info("Reported BPM: " + str(bpm))

            except Exception as e:
                LOGGER.exception("Couldn't acquier BPM for well " + str(video_metadata['well_id'])
                                    + " in loop " +
                                    str(video_metadata['loop'])
                                    + " with channel " + str(video_metadata['channel']))
            finally:
                well_result['well_id']  = video_metadata['well_id']
                well_result['loop']     = video_metadata['loop']
                well_result['channel']  = video_metadata['channel']
                well_result['bpm']      = bpm
                well_result['fps']      = fps
                
                # qc_attributes may help in dev to improve the algorithm, but are unwanted in production.
                if args.debug:
                    well_result.update(qc_attributes)

                results = results.append(well_result, ignore_index=True)

                gc.collect()

    except Exception as e:
        LOGGER.exception("Couldn't finish analysis")

    return results

# Run algorithm on a single well
def run_algorithm(well_frame_paths, video_metadata, args, resulting_dict_from_crop):
    LOGGER.info("Analysing video - "
                + "Channel: " + str(video_metadata['channel'])
                + " Loop: " + str(video_metadata['loop'])
                + " Well: " + str(video_metadata['well_id'])
                )

    # TODO: I/O is very slow, 1 video ~500mb ~20s locally. Buffer video loading for single machine?
    # Load video
    video_metadata['timestamps'] = io_operations.extract_timestamps(
        well_frame_paths)

    # TODO: Move the cropping out of here. 
    # This does not overlap with analysis and should therefore be in it's own function

    # Crop and analyse
    if args.crop == True and args.crop_and_save == False:
        LOGGER.info(
            "Cropping images - NOT saving cropped images")
        # We only need 8 bits video as no images will be saved
        video8 = io_operations.load_well_video_8bits(
            well_frame_paths)
        # now calculate position based on first 5 frames 8 bits
        embryo_coordinates = segment_heart.embryo_detection(
            video8[0:5])  # get the first 5 frames
        # crop and do not save, just return 8 bits cropped video
        video, resulting_dict_from_crop = segment_heart.crop_2(
            video8, args, embryo_coordinates, resulting_dict_from_crop, video_metadata)
        # save panel for crop checking
        io_operations.save_panel(resulting_dict_from_crop, args)

    elif args.crop_and_save == True:
        LOGGER.info("Cropping images and saving them...")
        # first 5 frames to calculate embryo coordinates
        video8 = io_operations.load_well_video_8bits(
            well_frame_paths, max_frames=5)
        # we need every image as 16 bits to crop based on video8 coordinates
        video = io_operations.load_well_video_16bits(well_frame_paths)
        embryo_coordinates = segment_heart.embryo_detection(video8)
        _, resulting_dict_from_crop = segment_heart.crop_2(
            video, args, embryo_coordinates, resulting_dict_from_crop, video_metadata)

        # now we need every frame in 8bits to run bpm
        video = io_operations.load_well_video_8bits(
            well_frame_paths)
        # save cropped images
        io_operations.save_cropped(video, args, well_frame_paths)
        # save panel for crop checking
        io_operations.save_panel(resulting_dict_from_crop, args)

    else:
        video = io_operations.load_well_video_8bits(
            well_frame_paths)

    bpm, fps, qc_attributes = segment_heart.run(video, vars(args), video_metadata)

    return bpm, fps, qc_attributes

def run_multifolder(args, dirs):
    # processes to be dispatched
    cmd_list = []
    procs_list = []

    print("### Directories to be analysed: ")
    # loop throw the folders
    for idx, path in enumerate(dirs):
        print(str(idx) + ": " + path)
        # get the indir and outdir arguments on the fly
        args.indir = path

        # get arguments for recursive call
        arguments_variable = [
            ['--' + key, str(value)] for key, value in vars(args).items() if value and value is not True]
        arguments_bool = ['--' + key for key,
                          value in vars(args).items() if value is True]
        arguments = sum(arguments_variable, arguments_bool)

        # absolute filepath and sys.executeable for windows compatibility
        filename = os.path.abspath(__file__)
        python_cmd = [sys.executable, filename] + arguments
        cmd_list.append(python_cmd)

    if args.cluster:
        for cmd in cmd_list:
            subprocess.run(cmd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
    else:
        max_subprocesses = 2
        print("Processing subfolders " + str(max_subprocesses) + " at a time.")
        i = max_subprocesses
        for cmd in cmd_list:
            p = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            procs_list.append(p)

            experiment_name = cmd[cmd.index("--indir")+1]
            experiment_name = os.path.basename(
                os.path.normpath(experiment_name))
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
    experiment_id, args = setup.process_arguments(args)
    setup.config_logger(args.outdir, ("logfile_" + experiment_id + ".log"), args.debug)

    LOGGER.info("Program started with the following arguments: " +
                str(sys.argv[1:]))

    ################################## MAIN PROGRAM START ##################################
    LOGGER.info("##### MedakaBPM #####")

    nr_of_videos, channels, loops = io_operations.extract_data(args.indir)
    if args.channels:
        channels = list(args.channels.intersection(channels))
        channels.sort()
    if args.loops:
        loops = list(args.loops.intersection(loops))
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
        main_directory = os.path.dirname(os.path.abspath(__file__))

        LOGGER.info("Running on cluster")
        try:
            job_ids = []
            for channel in channels:
                for loop in loops:
                    LOGGER.info("Dispatching wells from " + channel + " " + loop + " to cluster")

                    # Prepare arguments to pass to bsub job
                    args.channels = channel
                    args.loops = loop

                    arguments_variable = [
                        ['--' + key, str(value)] for key, value in vars(args).items() if value and value is not True]
                    arguments_bool = ['--' + key for key,
                                      value in vars(args).items() if value is True]
                    arguments = sum(arguments_variable, arguments_bool)

                    exe_path = os.path.join(main_directory, 'cluster.py')
                    # pass arguments down. Add Jobindex to assign cluster instances to specific wells.
                    python_cmd = ['python3', exe_path] + arguments + ['-x', '\$LSB_JOBINDEX']

                    jobname = 'heartRate' + args.wells + str(args.maxjobs)

                    bsub_cmd = ['bsub', '-J', jobname, '-M40000', '-R', 'rusage[mem=16000]']

                    if args.email == False:
                        if args.debug:
                            outfile = os.path.join(
                                args.outdir, 'bsub_out/', r'%J_%I-outfile.log')
                            os.makedirs(os.path.join(
                                args.outdir, 'bsub_out/'), exist_ok=True)
                            bsub_cmd += ['-o', outfile]
                        else:
                            bsub_cmd += ['-o', '/dev/null']

                    cmd = bsub_cmd + python_cmd  # calling source medaka_env was throwing a error

                    # Create a job array for each well
                    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

                    LOGGER.debug(cmd)

                    stdout_return = result.stdout.decode('utf-8')
                    LOGGER.info("\n" + stdout_return)

                    # Get jobId for consolidate command later
                    i1 = stdout_return.find('<') + 1
                    i2 = stdout_return.find('>')
                    job_ids.append(stdout_return[i1:i2])

            # Create a dependent job for final report
            job_ids = [("ended(" + s + ")") for s in job_ids]
            w_condition = '&&'.join(job_ids)

            # assign unique name.
            # Target completion of all experiment analysis with ended(HRConsolidate-*).
            # Needed in test_accuracy when JOB_DEP_LAST_SUB = 1 in lsb.params.
            unique_job_name = "HRConsolidate-" + str(job_ids[0])
            consolidate_cmd = ['bsub', '-J', unique_job_name, '-w', w_condition, '-M3000', '-R', 'rusage[mem=3000]']

            if args.email == False:
                if args.debug:
                    outfile = os.path.join(
                        args.outdir, 'bsub_out/', r'%J_consolidate.log')
                    os.makedirs(os.path.join(
                        args.outdir, 'bsub_out/'), exist_ok=True)
                    consolidate_cmd += ['-o', outfile]
                else:
                    consolidate_cmd += ['-o', '/dev/null']

            tmp_dir = os.path.join(args.outdir, 'tmp')
            exe_path = os.path.join(main_directory, 'src/', 'cluster_consolidate.py')
            python_cmd = ['python3', exe_path, '-i', tmp_dir, '-o', args.outdir]

            # consolidate_cmd += ['source', 'activate', 'medaka_env', '&&']  # calling source medaka_env here was throwing a error
            consolidate_cmd += python_cmd

            LOGGER.debug(consolidate_cmd)
            subprocess.run(consolidate_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        except Exception as e:
            LOGGER.exception("During dispatching of jobs onto the cluster")

    elif args.only_crop == False:
        LOGGER.info("Running on a single machine")

        results = analyse(args, channels, loops)

        ################################## OUTPUT ##################################
        LOGGER.info("#######################")
        LOGGER.info("Finished analysis")
        nr_of_results = len(results)
        if (nr_of_videos != nr_of_results):
            LOGGER.warning("Logic fault. Number of results (" + str(nr_of_results) +
                           ") doesn't match number of videos detected (" + str(nr_of_videos) + ")")

        results['version'] = SOFTWARE_VERSION
        io_operations.write_to_spreadsheet(args.outdir, results, experiment_id)

    else:
        LOGGER.info("Only cropping, script will not run BPM analyses")

        resulting_dict_from_crop = {}
        for well_frame_paths, video_metadata in io_operations.well_video_generator(args.indir, channels, loops):

            LOGGER.info("Looking at video - "
                        + "Channel: " + str(video_metadata['channel'])
                        + " Loop: " + str(video_metadata['loop'])
                        + " Well: " + str(video_metadata['well_id'])
                        )

            # we only need the first 5 frames to get position averages
            video8 = io_operations.load_well_video_8bits(
                well_frame_paths, max_frames=5)
            # we need every image as 16 bits to crop based on video8 coordinates
            video16 = io_operations.load_well_video_16bits(well_frame_paths)
            embryo_coordinates = segment_heart.embryo_detection(video8)

            # print("embryo")
            # print(embryo_coordinates)

            _, resulting_dict_from_crop = segment_heart.crop_2(
                video16, args, embryo_coordinates, resulting_dict_from_crop, video_metadata)
            # save cropped images
            io_operations.save_cropped(video16, args, well_frame_paths)
            # save panel for crop checking
            io_operations.save_panel(resulting_dict_from_crop, args)
            # here finish the script as we only need is save the cropped images


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
