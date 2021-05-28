### Script modified, based on Jack's script (https://github.com/monahanj/medaka_embryo_heartRate).

The main script file (in python) is medaka_embryo_heartRate.py. The sh files are wrappers to submit the jobs sequentially or ate the same time as an array_like job (if on the cluster). When running on the cluster, each sub-job would be one well on the plate.

Use the yml file to create your conda env with all necessary packs.

Upoload all the 4 files (medaka_embryo_heartRate.py, consolidated.py, analyse_plate_array_run.sh, and analyse_plate_array.sh) to a same folder on the cluster.

You don't need to activate the medaka_env or create an interactive job on the cluster, as the script takes care of everything.

# Usage examples in a single server:

## example 1 (running all possible wells in specified loops):

bash analyse_plate_array.sh -i /absolute_path/200706142144_OlE_HR_IPF0_21C/croppedRAWTiff -o /any_absolute_path/reports/ -l LO001.LO002 -c True -a 98 -e True -p 8 -b False

* in this example, the argument -w (wells) is NOT necessary

## example 2 (running a sequence of wells in specified loops):

bash analyse_plate_array.sh -i /absolute_path/200706142144_OlE_HR_IPF0_21C/croppedRAWTiff -o /any_absolute_path/reports/ -l LO002 -w [1,2,3,10] -c True -a 98 -e True -p 8 -b False

## example 3 (running a range of wells in specified loops):

bash analyse_plate_array.sh -i /absolute_path/200706142144_OlE_HR_IPF0_21C/croppedRAWTiff -o /any_absolute_path/reports/ -l LO002 -w [10-21] -c True -a 98 -e True -p 8 -b False

#### Observe that in single server mode, it is not possible to use a mix betwen sequences and range of wells in the same command, you must use one or another. However, you can run the bash script two times, one for a range, other for a sequence, but results will be saved in different date-time-based folders. As an option, you can, when running aditional commands, supress the creation of an date and time folder in the output path, and pointing the outputh path to the previsoly created folder. For supressing creation of date and time folder, use the parametter -b True

## Explanation of the arguments:

-i -> the input folder. This path is the folder in which all the images are. Inside this folder must be the images files for every well and every loop. The script will identify the type of the image by itself (tiff or JPEG).

-o -> the output folder. This path is where the script will create a time-date named folder with all the results. After the analyses finish, you can rename the folder to any name you want.

-l -> the loops you to be read. They need to be dot-separated and with no spaces. For example, -l LO001.LO002.LO003

-w -> optional parameter. You can optionally specify the wells to read. It can be sequence o numbers, comma-separated, and/or a range of numbers hyphen-separated. Do not use spaces. E.g.: [1,3,4,10-20]. If insering overlapping values, it will throw an error.

-c -> optional parameter (Default is False). If the script needs to crop the image, set it to True. It is useful if you have not previously cropped the images so that the script will crop them on the fly. Note that nothing new will be created; the script will discard cropped files after analyses. If you try to run the script without cropping, the script will probably fail after a long time analyzing each well, as analyzing full dimensions images is memory consuming.

-a -> optional parameter. After a first screening of the data, you can optionally rerun the script and insert the expected average of your data. It is useful as the script sometimes detects more than one furrier peak in a few cases, especially if the two heart chambers were segmented. The script can't identify which peak is correct without any reference, and sometimes the wrong peak is chosen, resulting in outliers. Then, if you know the distribution of your data, you can use this argument to decrease the outliers in your results, as it helps the algorithm identify the correct peak.

-e -> Optional parameter. For debugging purpose, you can set it to True to receive emails with the results of each well (subjob). The email address will be the email of the user logged on the cluster.

-p -> optional parameter. If set to an integer number, the script will open the specified number of process when running the slow mode. The slow mode runs automatically when the standard script fails. It takes action to try to detect the heart rate, but it can take a long time if running in just one or a few processors PC. Ideally, the number of processes to open should be the same number of virtual processors in your PC. When you run the script, it can count and inform you wheter you are using the ideal number of processors or not.

-b -> optional argument. Supress the creation of a date and time folder in the output dir. Set False to create it, True to NOT create it. May be usefull to save results together the results of a previous analyses and run a final report at the end.

# Usage in a farm of servers (cluster, LSF):

In this mode, it is possible to run all the weels and loops simultaneously, depending on the cluster availability. If all weels and loops can run simultaneously, finishing the whole plate with several loops can take as few as 15-20 minutes.

The arguments for this analyses are the same as above, but you must use the argument -f True to indicate that you are using a farmer of servers (cluster). Another difference is that in this mode, different from the single server mode, you can use the two types of wells selection combined, e.g. [10,11,13,30-40]. Note that the well´s selection (if it is the case) will be the same for each loop requested. And it is important to note that in this case, the LSF system will open one job_ID for each loop. It is important to know as you may need to kill or requeue jobs after submitting them.

In this mode, you can use the parameter -m [integer] to define the maximum number of jobs to run at the same time. It is helpful on busy servers if you don't want to bother others users. E.g.: -m 30

# What happens after a job is submitted?

If single server mode is used, the script will read one well at a time. Reading a whole plate can take dozens of hours, depending on the computer performance.

If in cluster mode, after running the script as a command line, it will generate an array of sub-jobs (all of them with the same id if using one Loop) on the cluster, and every well will be read almost at the same time with a different subJob identifier. Note that it is only the case if you do not specify weels. If you specify wells, the script will open one job_id for each loop, and subjobs for each well. 

If there are not enough hosts available, some sub-jobs will show PENDING, waiting for free hosts. About 15-20 minutes later, a full plate with two loops can be finished, for example (may vary, depending on the host's availability). 

Note that for each loop you submit, and if you do not specify the wells, 96 sub-jobs will be created, even if you don't need to read the whole plate. In this case, the empty wells (with no images) will fail without causing any problem to analyses.

One job with a different Id will be created, and it will be Pending status until any other job finish. It is a conditional job and is responsible for making a final report in CSV format.

Two sets of report files (JPEG and CSV files) will be created, one inside each loop folder for that specific loop and another (inside the main folder) for all loops merged in the same file.

To see the jobs running and job status, use \<bjobs\>
  
After some time, if any sub-job is stuck running in a busy host, you can reschedule the job to another host using the command \<breschedule Job_Id\>. It will reschedule all running jobs with the specified ID to another available host (remember that all jobs have the same ID, but different loops have different jobs_ID). It only works for stuck running jobs; if the subjob is pending status, you should wait, as any host with the requirements is not available yet, and the job is on the queue. 

  
For any reason, you can kill all sub-jobs at the same time with the command \<bkill Job_Id\>
Remember that if you have more than one loop, you will have one job_id for each loop. Then you have two options: or kill every Job_id one at a time, or kill all jobs related to the Heart Rate script at the cluster, using the job name, as the example: bkill -J heartRate
  
# How to generate the final report
If you are using the farm mode (or cluster mode), the final report will be generated automatically. As explained earlier, one of the jobs will keep pending untill any other job finish, to run the final report. The CSV main report will be generated from each CSV file existent inside each folder in main folder. If for any reason, any of the CSV is lacking inside subfolders in main folder, the script will fail.

If you are using the single server mode and you want a final report, you must run the python script manually. e.g.:
python3 consolidated.py -i [path for the main dir, where loops folders with results are] -o [output dir, can be the same as indir] 
This script will go throw every subfolder in main folder, looking for CSV files, and will generate a distribution graph and CSV file with all results merged in a single file.

![final_graph](https://user-images.githubusercontent.com/6963691/119535040-97b4a400-bd55-11eb-95f0-947dacc85e73.jpg)

