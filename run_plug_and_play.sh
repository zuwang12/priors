#!/bin/bash

interval=20
num_cities=100
now=$(date +"%F_%T")
run_name="tsp${num_cities}_plug_play_${now}"
constraint_type="path"

mkdir -p "./logs/${constraint_type}/${run_name}"

available_devices=(0 1 2 3) 
num_devices=${#available_devices[@]}

for (( start_idx=0; start_idx+interval<=1280; start_idx+=interval )); do
    end_idx=$((start_idx + interval))
    
    # Set CUDA device
    device_number=${available_devices[$(( (start_idx / interval) % num_devices ))]}
    CUDA_VISIBLE_DEVICES=$device_number nohup python inference_result.py --run_name $run_name --start_idx $start_idx --end_idx $end_idx --num_cities $num_cities --constraint_type $constraint_type \
        > "./logs/${constraint_type}/${run_name}/from${start_idx}_to${end_idx}.log" 2>&1 &
done
