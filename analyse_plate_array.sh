#!/bin/bash
source activate medaka_env

# for details, visit https://github.com/marciosferreira/medaka_embryo_heartRate_modified" 

unset indir
unset out_dir
unset loops
unset folders
unset farm
unset crop
unset maxjobs
unset average
unset email
unset wells
unset process
unset supress_dtf

basedir=$PWD

while getopts ":i:o:l:f:c:m:a:e:w:p:b:" opt; do
        case ${opt} in
                i) indir=$OPTARG ;;
                o) out_dir=$OPTARG ;;
                l) loops=$OPTARG ;;                
                f) farm=$OPTARG ;;
                c) crop=$OPTARG ;;
                m) maxjobs=$OPTARG ;;
				a) average=$OPTARG ;;
				e) email=$OPTARG ;;
				w) wells=$OPTARG ;;
				p) process=$OPTARG ;;
				b) supress_dtf=$OPTARG ;;
                *)
                        echo 'Error in command line parsing' >&2
                        echo "echo "if you need, visit https://github.com/marciosferreira/medaka_embryo_heartRate_modified for instructions and troubleshooting!" >&2"
                        echo ""
						echo ""

                        exit 1
        esac
done

shift "$(( OPTIND - 1 ))"

if [ -z "$loops" ] ; then
        echo " " >&2
        echo 'Missing -l mandatory argument. Insert all loops to read.' >&2
        echo 'Insert all loops names dot (.) separated with no spaces betwen them' >&2
        echo "echo "if you need, visit https://github.com/marciosferreira/medaka_embryo_heartRate_modified for instructions and troubleshooting!" >&2"
        echo ""
		echo ""
        exit 1
fi


if [ -z "$out_dir" ] ; then
        echo " " >&2
        echo "Error: Missing -o mandatory argument. Specify a foldfer to where the script will create a date folder and save results"
        echo "echo "if you need, visit https://github.com/marciosferreira/medaka_embryo_heartRate_modified for instructions and troubleshooting!" >&2"
        echo ""
		echo ""
        exit 1     

fi



if [ -z "$indir" ] ; then
        echo " " >&2
        echo 'Error: Missing -i mandatory argument. Please specify the absoluthe path for where the images are' >&2
        echo ""
		echo ""
        exit 1
fi



echo ""
echo ""
echo ""
echo "#####################################################################"
echo "MEDAKA HEART RATE DETECTION SCRIPT"
echo  "Visit https://github.com/marciosferreira/medaka_embryo_heartRate_modified for instructions and troubleshooting!" >&2
echo "#####################################################################"


#split and count loops and wells list
IFS='.'
read -a strarr_l <<< "$loops"
loops_quantity=${#strarr_l[*]}
IFS=' '


if [ -z "$wells" ] ; then
	echo ""
	has_wells=0
	echo " "
	echo "Reading all 96 wells for each requested loop(s)"
else
	has_wells=1
	echo " "
	echo "Reading only the well(s) $wells in the requested loop(s)" >&2

fi



# verify if path have image folders
files_tif=( ${indir}/*".tif" )
files_jpg=( ${indir}/*".jpg" )
files_tif_number=${#files_tif[@]}
files_jpg_number=${#files_jpg[@]}
sum=$(( $files_tif_number + $files_jpg_number ))
if [ ${sum} -gt 50 ]; then 
    echo "-------"
    echo "A total of ${sum} image(s) file(s) were detected in in_dir" >&2
    echo "-------"
else
    echo "-------"
    echo "Error: Zero or not enought image files detected in indir folder. Please verify the path." >&2
    echo "if you need, visit https://github.com/marciosferreira/medaka_embryo_heartRate_modified for instructions and troubleshooting!" >&2
    echo ""
    echo ""
    exit 1

fi


#verify if loops are in in_dir and exit if any loop are not in in_dir
for val in "${strarr_l[@]}";
do
  #printf "$val\n"
  files_loop=( ${indir}/*"$val"* )
  files_loop_number=${#files_loop[@]}
  
  if [ ${files_loop_number} -gt 1 ]; then
      echo "The requested loop $val was detected in in_dir" 
  else
      echo "Error: NO files with $val in file name was detected in in_dir. Please verify and try again!".
      echo "if you need, visit https://github.com/marciosferreira/medaka_embryo_heartRate_modified for instructions and troubleshooting!" >&2
      echo ""
      echo ""
      exit 1
  fi

  
done

#only echoing usefull informations
echo "-------"
echo "loops quantity: $loops_quantity"




# verify if output path exists and create date folder for output

if [[ -d $out_dir ]] ; then
	if [ -z "$supress_dtf" ] || [ "$supress_dtf" != True ] ; then		
		#append date and time to the output directory
		date_and_time=$(date '+%Y-%m-%d_%H%M%S')
		out_dir=${out_dir}/MDK_${date_and_time}
		mkdir -p ${out_dir}
	fi
else
 	echo "-------"
	echo "The output path was NOT located in your system. Please verify or created it first"
	echo "if you need, visit https://github.com/marciosferreira/medaka_embryo_heartRate_modified for instructions and troubleshooting!" >&2
	echo ""
	echo ""
    	exit 1
fi

#deal with crop argument
echo "-------"
if [ "$crop" != "True" ] ; then    
    echo "The pictures will NOT be cropped. If the images are of high resolution, it may cause error of memory exhaustion" >&2
    crop="False"
else
    echo "The pictures will be cropped and they will be discarded after analyses" >&2
fi


#deal with email argument
if [ "$email" == "True" ] ; then
    echo "-------"    
    echo 'email with analyses results will be sent (to the cluster logged in user) for debugging purposed' >&2
    email=""
else
    email="-o /dev/null"
fi



if [ "$has_wells" -gt 0 ] ; then	
	reads=$wells
else
	reads=[1-$(expr 96 \* $loops_quantity)]
	echo "requested reads: $reads"
	echo "-------"
	echo "The script will lauch one read (subjob) for each well in a plate (multiplied by the loops quantity), even if the plate is not full (and the images files are not present)"
	echo "Empty wells will fail, without prejudice for the whole analyses"
	echo "-------"

fi


echo "Results will be saved at: $out_dir" >&2

echo "-------"


#prepare the bash command line for argument max-jobs
if [ -z "$maxjobs" ] ; then
    maxjobs=''
else
    echo "Max $maxjobs jobs allowed at the same time"
    maxjobs=%$maxjobs
    echo "-------"    
fi


#deal with average argument
if [ -z "$average" ] ; then
    average='0'
else
    echo "In case of two peaks detected, the algorithm will discard the outlier based on the average of $average"       
fi


#deal with process argument
if [ -z "$process" ] ; then
    process=1
fi



if [[ $crop != 'True' ]] ; then
	crop="--no-crop"	
else
	crop="--crop"	
fi


echo "Number of process to run simultaneously in case of script needs to run the slow mode: $process" >&2
echo "*limitated by the number of processors of your computer."
echo "This feature makes the script faster in a few cases in which the original method fails,"
echo "and then the script automatically tries to run an alternative method (slow method) to abtain the heart rate"

echo "-------"



if [ "$farm" == "True" ] ; then

	
	echo "Set to cluster mode (farm of computers; LSF)"
	echo ""
	echo ""
	echo "#####################################################################"
     
	for j in ${strarr_l[@]}; do
		#Create a job array for each well and each loop
		bsub -J "heartRate${reads}${maxjobs}" -M20000 -R rusage[mem=8000] $email bash analyse_plate_array_run.sh $indir $j $out_dir $crop $average $process # -o /dev/null depois do ] -o log_array.txt or -o /dev/null

	done

	#Create a dependent job for final report
	bsub -J "consolidated" -w "ended(heartRate)"  -M3000 -R rusage[mem=3000] $email python3 consolidated.py -i $out_dir -o $out_dir #-o log_consolidated.txt

else
	farm=False

	
	echo "Set to single server mode"
	echo ""
	echo ""
	echo "#####################################################################"



    
	if [ "$has_wells" -gt 0 ] ; then


		

		if [[ "$wells" == *","* ]] && [[ "$wells" == *"-"* ]] ; then
			
			echo " "
	  		echo "ERROR: In single server mode, the script only accept a range of values (e.g.: [10-20] or coma separated values (e.g.: 2,6,8)." 
	  		echo "It is not possible to mix these options together in this version of the script."
	  		echo "Observe that if you enter more than one loop, the script will read the same sequence of wells for every loop"
	  		echo " "
	  		echo " "
	  		exit 1


		elif [[ "$wells" == *","* ]]; then
			
			wells="${wells:1:-1}"
			IFS=','
			read -a strarr_weels <<< "$wells"
			IFS=' '

			for i in ${strarr_weels[@]}; do
				if [[ $i -gt 96 ]] ; then
					echo "ERROR: It is not possible to have well numbers higher than 96"
					exit 1
				fi

				if [[ $i -lt 1 ]] ; then
					echo "ERROR: It is not possible to have well numbers lower than 1"
					exit 1
				fi

			done

			#for each_well in "${strarr_weels[@]}"; do
				#echo $each_well
			#done


			for i in ${strarr_weels[@]}; do

				for j in ${strarr_l[@]}; do


					python3 segment_heart.py -i $indir -l $j $crop -o $out_dir -ix $i -a $average -p $process
		       
				done

			done


		elif [[ "$wells" == *"-"* ]]; then

			wells="${wells:1:-1}"
			IFS='-'
			read -a strarr_weels <<< "$wells"
			IFS=' '
					
			first_well=${strarr_weels[@]:0:1}
			if [[ $first_well -gt 95 ]] ; then
				echo "ERROR: The first well can not be higher than 95"
				exit 1
			fi

			last_well=${strarr_weels[@]:1:2}
			if [[ $last_well -gt 96 ]] ; then
				echo "ERROR: The last well can not be higher than 96"
				exit 1
			fi


			if [[ $last_well -lt $first_well ]] ; then
			echo "ERROR: first well must be lower than last well"
			exit 1
			fi


			for ((i=${first_well};i<=${last_well};i++)); do

				for j in ${strarr_l[@]}; do


					python3 segment_heart.py -i $indir -l $j $crop -o $out_dir -ix $i -a $average -p $process
		       
				done

			done

		else
			wells="${wells:1:-1}"
			for j in ${strarr_l[@]}; do
					python3 segment_heart.py -i $indir -l $j $crop -o $out_dir -ix $wells -a $average -p $process		       
			done
			


        fi	

	else 

	
	#Iterate over all 96 wells in a plate, for each loop in -l argument
		for i in 0{1..9} {10..96}; do

			for j in ${strarr_l[@]}; do				
				python3 segment_heart.py -i $indir -l $j $crop -o $out_dir -ix $i -a $average -p $process
		       
			done

		done

	fi


fi