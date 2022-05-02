in_dir="/Users/anirudh/Documents/HiWi-22/MEDAKA/experiments/images/24FPS_F2_testsets/F2_24FPS_210719155352_OlE_HR_IPF2_2x_BQ_21C/Video-"

out_dir="/Users/anirudh/Documents/HiWi-22/MEDAKA/medaka_bpm/tmp"

annot_file="/Users/anirudh/Documents/HiWi-22/MEDAKA/experiments/images/24FPS_F2_testsets/F2_24FPS_210719155352_OlE_HR_IPF2_2x_BQ_21C/Annotations/Video-"

annot_ext="json"

for i in {1..100}
    do
        if [ -f $annot_file$i"."$annot_ext ]; then
            python annotation_compare.py -i $in_dir$i -o $out_dir -af $annot_file$i"."$annot_ext
            echo "Compared annotation-${i}."
        fi
    done