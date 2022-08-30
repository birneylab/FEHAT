### Script modified, based on Jack's script (https://github.com/monahanj/medaka_embryo_heartRate).

The main script file is medaka_bpm.py

Use the yml file to create a conda env with all necessary pkgs.

The software can be run on a single machine or on an lsf cluster. When running in cluster mode, busb is used to create jobs and distribute data.

# Usage examples in a single server:

## example 1 (running everything in the input directoy in single machine):

	(medaka_bpm)$ python medaka_bpm.py -i /absolute_path/200706142144_OlE_HR_IPF0_21C/ -o /any_absolute_path/reports/

## example 2 (running everything in the input directory on the cluster):

	(medaka_bpm)$ python medaka_bpm.py -i /absolute_path/200706142144_OlE_HR_IPF0_21C/ -o /any_absolute_path/reports/ --cluster 

## example 3 (running specified loops, channels and well):

	(medaka_bpm)$ python medaka_bpm.py -i /absolute_path/200706142144_OlE_HR_IPF0_21C/ -o /any_absolute_path/reports/ -l LO002.LO001 -c CO3.CO6 -w WE00001.WE00002

## Explanation of the arguments:

**Necessary arguments:**

-i, --indir

The input folder. This path is the folder in which all the images are. Inside this folder must be the images files for every well and every loop.
Images must be in tif format.

-o, --outdir 

The output folder. This path is where the script will write the results.

**Optional arguments:**

-l, --loops

Restriction on the loops to read. They need to be dot-separated and with no spaces.
For example, ``-l LO001.LO002.LO003``

-c, --channels

Restriction on the channels to read. They need to be dot-separated and with no spaces, like the loops argument. 
For example, ``-c CO6.CO4.CO1``

-w, --wells

Restriction on the wells to read. They need to be dot-separated and with no spaces, like the loops argument. Works only in single machine mode (does not work on cluster).
For example, ``-w WE00001.WE00002``

-f, --fps

If not provided, framerate is automatically calculated from image timestamps.
Alternatively, set the framerate through this argument.

-m, --maxjobs

For cluster mode, defines the maximum number of jobs to run at the same time. 
It is helpful on busy servers if you don't want to bother others users. E.g.: ``-m 30``

--email

For debugging purpose, you can set this argument to receive emails with the results of each well (subjob). 
The email address will be the email of the user logged on the cluster.

--crop

Use it if the script needs to crop the image. It is useful if you have not previously cropped the images so that the script will crop them on the fly. 
Note that nothing new will be created; the script will discard cropped files after analyses. If you need keep the cropped files, see the crop_and_save option bellow. If you try to run the script without cropping, the script will probably fail after a long time analyzing each well, as analyzing full dimensions images is memory consuming. You can use the parameter -s (bellow) to adjust the cropping. If you are not sure if the cropping offset is ok, we suggest using the option "only_crop" for some wells just to see if is being cropped in the right position. 

--only_crop

Only crop images (not run bpm script) based on indir and save croped images and a resulting panel report (a ".png" file) in outdir. If there are multiple folders in indir, the script will try to crop images in every folder and save them in different folders. An image called offset_verifying.png will be created at the beginning of the analyses, so if you want, you can stop the script and see if the parameter -s needs to be adjusted.

--crop_and_save

The script will crop images (apart of running bpm script), and save them. This is useful because the bottleneck of the script is reading and writing images, then, using this option, reading and writing will be done just once for bpm and cropping, saving a lot of time. If there are multiple folders in indir, the script will try to crop images in every folder and save them in different folders. An image called offset_verifying.png will be created at the beginning of the analyses, so if you want, you can stop the script and see if the parameter -s needs to be adjusted.

-s

Only useful when cropping images. It is the size of the expected radius of the embryo, and this value will be used for cropping based on embryo's center of mass. The default value is 300 px. If you don´t know how much to use, we suggest test first using the option only_crop for some wells, then stop the script and check the offset.


# What happens after a job is submitted?

Two sets of report files (JPEG and CSV files) will be created, one inside each loop folder for that specific loop and another (inside the main folder) for all loops merged in the same file.

  
# Notes on usage in a farm of servers (cluster, LSF):

In this mode, it is possible to run all the weels and loops simultaneously, depending on the cluster availability. 
If all weels and loops can run simultaneously, finishing the whole plate with several loops can take as few as 15-20 minutes.

Results and individual log files are stored in the /tmp folder in the output directory and merged afterwards.

The arguments for this analyses are the same as above, but you must use the argument 
``--cluster`` to indicate that you are using a farmer of servers (cluster). 

It is important to note that in this case, the LSF system will open one job_ID for each loop in each channel

After running the script as a command line, it will generate an array of sub-jobs (all of them with the same id if using one Loop)
On the cluster, and every well will be read almost at the same time with a different subJob identifier. 
Note that it is only the case if you do not specify weels. 
If you specify wells, the script will open one job_id for each loop, and subjobs for each well. 

If there are not enough hosts available, some sub-jobs will show PENDING, waiting for free hosts. About 15-20 minutes later, a full plate with two loops can be finished, for example (may vary, depending on the host's availability). 

Note that for each loop you submit, and if you do not specify the wells, 96 sub-jobs will be created, even if you don't need to read the whole plate. In this case, the empty wells (with no images) will fail without causing any problem to analyses.

One job with a different Id will be created, and it will be Pending status until any other job finish. It is a conditional job and is responsible for making a final report in CSV format.

To see the jobs running and job status, use ``bjobs``

After some time, if any sub-job is stuck running in a busy host, you can reschedule the job to another host using the command ``breschedule Job_Id``.

It will reschedule all running jobs with the specified ID to another available host (remember that all jobs have the same ID, but different loops have different jobs_ID). 
It only works for stuck running jobs. 
If the subjob is pending status, you should wait, as any host with the requirements is not available yet, and the job is on the queue. 

For any reason, you can kill all sub-jobs at the same time with the command ``bkill Job_Id``.

For example: ``bkill -J heartRate``

Remember that if you have more than one loop, you will have one job_id for each loop.
Then you have two options: or kill every Job_id one at a time, or kill all jobs related to the Heart Rate script at the cluster, using the job name.

# Notes on single machine analysis:
If single server mode is used, the script will read one well at a time. 

Reading a whole 96-well plate can take a few of hours.

Loading the images into memory is a major bottleneck at the moment.

# Benchmarking algorithm performance:
To assess accuracy and classification rate of a specific version of the algorithm, test_accuracy.py can be used.

It takes the same arguments as an input. The input directory should contain several folders with data to test upon. 
In addition, next to the folders, a ground truth file called "ground_truths.csv" hast to be placed.

Example folder structure:

	.../input-directory/
			├── Experiment-folder 1/
			|		├── frame.tiff
			|		├── frame.tiff
			|		└── [...]
			|
			├── Experiment-folder 2/
			|		├── frame.tiff
			|		├── frame.tiff
			|		└── [...]
			|
			└── ground_truths.csv

The ground truth file has to follow the following format. 
Matching is performed over all fields.
"groundtruth" field will be used to compare with BPM in the respective result csv files.

	| DATASET      			| Index			|	WellID		|	Loop		|	Channel	|	|	ground truth	|

	| ----------- 			| ----------- 	| ----------- 	| ----------- 	| ----------- 	| ------------ 		|
	| Experiment-folder 1   | 1       		| WE0001		| LO001			| CO6			| 104				|
	| Experiment-folder 1   | 2        		| WE0002		| LO001			| CO6			| 83				|
	| ...					| ...			| ...			| ...			|  ...			| ...				|
	| Experiment-folder 2   | 1        		| WE0001		| LO001			| CO6			| NA				|
	| Experiment-folder 2   | 2        		| WE0002		| LO001			| CO6			| 110				|
	| ...					| ...			| ...			| ...			|  ...			| ...				|
