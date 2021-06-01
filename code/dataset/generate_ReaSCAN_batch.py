# This script you can generate based on random seeds, and then combine in what ever way you would like.

import itertools
import os

command_pattern = "p3"
special_condition = "rd"
prefix = command_pattern
if special_condition != "":
    prefix = f"{command_pattern}-{special_condition}"
total_command = 4000
job_count = 50
per_job_command = (total_command // job_count)
if total_command%job_count != 0:
    assert False
print(f"starting per_job_command={per_job_command}")

for i in range(job_count):
    
    command = f'tmux new-session -d -s session{i} "python generate_ReaSCAN.py \
                --mode train \
                --n_command_struct {per_job_command} \
                --date 2021-05-30 \
                --grid_size 6 \
                --n_object_max 13 \
                --per_command_world_retry_max 500 \
                --per_command_world_target_count 180 \
                --output_dir ../../data-files-{prefix}/ReaSCAN-compositional-{prefix}-jobid-{i}/ \
                --include_random_distractor \
                --full_relation_probability 1.0 \
                --command_pattern {command_pattern} \
                --save_interal 50 \
                --seed {i}"'

    print(f"starting command-{i}")
    os.system(command)