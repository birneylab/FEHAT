# FEHAT - Fish Embryo Heartbeat Assessment Tool

FEHAT extracts the heart beats per minute from fish embryo videos - fully automated!

The software can be run on a single machine or on an HPC cluster running LSF. 
When running in cluster mode, busb is used to create jobs and distribute data. For integration into your own HPC environment, tweaking may be necesarry.

## Installation
1. Download or clone this repository:

        git clone https://github.com/birneylab/FEHAT.git
        cd FEHAT/

2. Setup a new Python virtual environment (optional):

        python -m venv fehat/
        source fehat/bin/activate

3. Install required packages:

        pip install -r requirements.txt

## Testing
A short 5 second example video can be found in `data/test_video/`. To test if everything was installed properly, run:

    python medaka_bpm.py -i data/test_video/ -o data/test_output

You should see the LOGGER printing on the console. Inside `data/test_output`, you should find a detailed results csv file and a logfile.
In addition, videos and images of the software analysis are saved for manual inspection.

## Usage examples
The software was designed to analyse either a single video, a collection of videos (referred to here as an "experiment") or a collection of experiments with a single run. Refer to the section on [input format](#input-format) for more details.

#### example 1. Run everything in the input directoy in single machine:

	python medaka_bpm.py -i <input_directory> -o <output_directory>

#### example 2. Run everything in the input directory on the HPC cluster:

	python medaka_bpm.py -i <input_directory> -o <output_directory> --cluster 

#### example 3. Run specific loops, channels or wells:

	python medaka_bpm.py -i <input_directory> -o <output_directory> -l LO001.LO002 -c CO3.CO6 -w WE00001.WE00002

## Input format
This software was designed to be used in conjunction with the *ACQUIFER Imaging Machine*. To integrate into you own experiment pipeline, you will need to conform to the input format or modify this application to suite your needs.

We can identify an individual given it's **well**, **loop**, **channel** and experiment id.
For a single experiment, a 96 well plate is filled with a single fish embryo in each well. The wells are then individually recorded one after another using a specific *channel*. This may be repeated for arbitrary many *loops*.
The experiment id is specified via the input folder's name.

### Video format
Videos are expected to be present as individual frames in .tiff format. See `data/test_video/` for an example.
Individual frame files are named in the following way:

    WE00037---D012--PO01--LO001--CO6--SL001--PX32500--PW0080--IN0010--TM280--X113563--Y038077--Z224252--T0015372985.tif

Where 
- `WE00037` indicated Well 37 on a 96 well plate and can range from ``WE00001`` to ``WE00096``.
- `LO001` indicates loop 1 and can range from `LO001` to  `LO999`.
- `CO6` indicates channel 6 and can range from `CO0` to  `CO9`.
- `SL001` indicates the frame number, i.e. this is the first frame of the video.
- `T0015372985` indicates the timestamp in milliseconds, at which the frame was recorded. In this case the image was aquired at timestamp 15372985ms. `T0000000000` refers to 0ms.

All other fields are metadata and need not be present. Videos that share a common value for well, loop and channel are then parsed as a single video.

### Directory structures
Input data can be organised in three different ways. A single video in a directory, an experiment directory (collection of videos) or multiple experiment directories.

Experiment directories should contain a uniqe identifier as a prefix, followed by an underscore:`<expeirment_id>_<my_experiment_name>`. This will get used as the experiment id during processing and output formatting. A valid name would for example be: `20230227_my_experiment`.

#### Case 1. A single video:

	.../Experiment_dir/
            ├── frame1.tiff
            ├── frame2.tiff
            └── [...]

#### Case 2. A single experiment. Contains multiple videos

	.../Experiment_dir/
            ├── video1_frame1.tiff
            ├── video1_frame2.tiff
            ├── [...]
            ├── video2_frame1.tiff
            ├── video2_frame2.tiff
            └── [...]

#### Case 3. Multiple experiment folders.
	.../input-directory/
			├── Experiment-folder 1/
			|		├── frame1.tiff
			|		├── frame2.tiff
			|		└── [...]
			├── Experiment-folder 2/
			|		├── frame1.tiff
			|		├── frame2.tiff
			|		└── [...]
            └── [...]

## Available CLI arguments

#### Necessary arguments

**-i**, **--indir** `PATH`

The input folder. See [input format](#input-format) for details.

**-o**, **--outdir** `PATH`

The output folder. This path is where the script will write the results. If none is given, creates a subfolder in input directory.

**Optional arguments:**

**-l**, **--loops**

Restriction on the loops to read. They need to be dot-separated and with no spaces.
For example, ``-l LO001.LO002.LO003``

**-c**, **--channels**

Restriction on the channels to read. They need to be dot-separated and with no spaces, like the loops argument. 
For example, ``-c CO6.CO4.CO1``

**-w**, **--wells**

Restriction on the wells to read. They need to be dot-separated and with no spaces, like the loops argument. Works only in single machine mode (does not work on cluster).
For example, ``-w WE00001.WE00002``

**-f**, **--fps**

If not provided, framerate is automatically calculated from image timestamps.
Alternatively, set the framerate through this argument.

**-m**, **--maxjobs**

For cluster mode, defines the maximum number of jobs to run at the same time. 
It is helpful on busy servers if you don't want to bother others users. E.g.: ``-m 30``

**--email**

For debugging purpose, you can set this argument to receive emails with the results of each well (subjob). 
The email address will be the email of the user logged on the cluster.

**--crop**

Crop mode. Videos will be read as for BPM analysis, but will be cropped instead. Useful to reduce the data load of raw images.

## Available config parameters
For more persistent adjustments to the software, we provide a `config.ini` config file.

    [DEFAULT]
    VERSION = v1.4
    MAX_PARALLEL_DIRS = 5
    DECISION_TREE_PATH = data/decision_tree.pkl

    [ANALYSIS]
    MIN_BPM = 70
    MAX_BPM = 310
    ARTIFICIAL_TIMESTAMPS = yes

    [CROPPING]
    EMBRYO_SIZE = 450
    BORDER_RATIO = 0.1

- **MIN_BPM**/**MAX_BPM**. Set a limit to which resulting BPM are still credible. Note that this potentially leads to loss of results, as uncredible values are thrown away.

- **MAX_PARALLEL_DIRS**. To facilitate faster processing on single machine mode, when analysing multiple experiment folders (case 3), experiment folders are analysed in parallel. You can adjust the MAX_PARALLEL_DIRS variable, to set a limit to how many are processed at the same time.

- **ARTIFICIAL_TIMESTAMPS**. If set to yes (default) will use equally spaced timestamps, according to given or estimated fps. If set to no, will attempt to use given timestamps of frames, but needs to interpolate pixel values and can be inaccurate.

- **EMBRYO_SIZE**. For cropping, assume a minimum embryo size that should not be cut out. Given in pixels.
- **BORDER_RATIO**. For cropping, crops at least this ratio around the edges of the images as it is assumed border for sure.

# Notes on single machine analysis:
If single server mode is used, the script will read one well at a time. 

Reading a whole 96-well plate can take a few of hours.

Loading the images into memory is a major bottleneck at the moment.

# Notes on usage in a farm of servers (cluster, LSF):

In this mode, it is possible to run all the wells and loops simultaneously, depending on the cluster availability. 
If all wells and loops can run simultaneously, finishing the whole plate with several loops can take as few as 15-20 minutes.

Results and individual log files are stored in the /tmp folder in the output directory and merged afterwards.

The arguments for this analyses are the same as above, but you must use the argument 
``--cluster`` to indicate that you are using a farmer of servers (cluster). 

It is important to note that in this case, the LSF system will open one job_ID for each loop and each channel

After running the script, it will generate an array of sub-jobs (all of them with the same id if using one Loop).

If there are not enough hosts available, some sub-jobs will show PENDING, waiting for free hosts.

Note that for each loop you submit, 96 sub-jobs will be created. In this case, the non-existing wells (with no images) will fail without causing any problem to the analysis.

One job with a different Id will be created, and it will be Pending status until any other job finish. It is a conditional job and is responsible for making a final report in CSV format.

To see the jobs running and job status, use ``bjobs``.

For any reason, you can kill all sub-jobs at the same time with the command ``bkill Job_Id``.

Or use the prefix of all job names to kill all related jobs: ``bkill -J heartRate``

Remember that if you have more than one loop, you will have one job_id for each loop.
Then you have two options: or kill every Job_id one at a time, or kill all jobs related to the Heart Rate script at the cluster, using the job name.

# Benchmarking algorithm performance:
To assess accuracy and classification rate of a specific version of the algorithm, test_accuracy.py can be used.

    python qc_analysis/test_accuracy.py -i <input_dir> -o <output_dir> 

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
"ground truth" field will be used to compare with BPM in the respective result csv files.

	| DATASET      			| Index			|	WellID		|	Loop		|	Channel	|	|	ground truth	|

	| ----------- 			| ----------- 	| ----------- 	| ----------- 	| ----------- 	| ------------ 		|
	| Experiment-folder 1   | 1       		| WE0001		| LO001			| CO6			| 104				|
	| Experiment-folder 1   | 2        		| WE0002		| LO001			| CO6			| 83				|
	| ...					| ...			| ...			| ...			|  ...			| ...				|
	| Experiment-folder 2   | 1        		| WE0001		| LO001			| CO6			| NA				|
	| Experiment-folder 2   | 2        		| WE0002		| LO001			| CO6			| 110				|
	| ...					| ...			| ...			| ...			|  ...			| ...				|

# Contributing to this repository
We welcome any reports on bug, usage experience and contributions in the form of bug patches and new features!
Feel free to open a thread in the issue tracker or contact any of the maintainers (Marcio Ferreira, Tomas Fitzgerald, Sebastian Stricker) directly.

## Did you find a bug?
Please open a thread in the github issue tracker.

## Did you write a patch that fixes a bug? Or a new feature? Or made any general improvements?
Great! We really appreciate any contributions.
- Please open a new GitHub pull request on the ```dev``` branch. 
- Make sure to commit about a single clearly defined issue/feature
- Please write a meaningful commit messages.
- Feel free to add yourself as a Contributor in the header of the file that you modified. If you add a new file, make sure to give it a header that is consistent with the rest of the repository.