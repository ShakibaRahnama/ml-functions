#!/bin/bash

glob_file="./video_groups.txt"
vid_dir="/cluster/projects/madanigroup/CLAIM/Colorectal/videos"
fra_dir="/cluster/projects/madanigroup/CLAIM/Colorectal/frames"
i=0
while read -r g;
do
    echo here
    out="output_${i}.out"
    err="error_${i}.err"
    job_file="job_${i}.job"
    wrap="python extract_allframes.py --video_dir ${vid_dir} --frame_dir ${fra_dir} --video_glob \"${g}\""
    job="#!/bin/bash
#SBATCH --mem=5G
#SBATCH -t 02:00:00
#SBATCH -p all
#SBATCH -c 1
#SBATCH -o ${out}
#SBATCH -e ${err}

${wrap}"
    echo "${job}" > ${job_file}
    sbatch ${job_file}
    echo there
    i=$((i + 1))
done < "$glob_file"
