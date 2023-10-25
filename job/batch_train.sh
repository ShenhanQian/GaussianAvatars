#!/bin/bash

# Define a list of subjects and sequences
# SUBJECTS=("055" "074" "104" "140" "165" "210" "218" "221" "238" "251" "253" "264" "302" "304" "306" "375")
# SUBJECTS=("055")  # tmp


# Loop through subjects and sequences and modify the command accordingly
GS_PORT=60040

for SUBJECT in "${SUBJECTS[@]}"; do
JOB_NAME="gs_${SUBJECT}_${GS_PORT}"

#======= train-300k =======#
# COMMAND="python train.py \
# -s $IKARUS/project/dynamic-head-avatars/code/multi-view-head-tracker/export/UNION_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_offsetS_whiteBg \
# -m $IKARUS/project/dynamic-head-avatars/code/gaussian-splatting/output/Union10EMOEXP_${SUBJECT}_eval_xyz-1e0_poseLr1e-5_exprLr1e-3_300k_densify5k-150k \
# --port $GS_PORT \
# --eval --white_background --bind_to_mesh \
# --lambda_xyz 1e0 \
# --iterations 300_000 --position_lr_max_steps 300_000 --densify_from_iter 5_000 --densify_until_iter 150_000
# "

#======= train-600k =======#
# COMMAND="python train.py \
# -s $IKARUS/project/dynamic-head-avatars/code/multi-view-head-tracker/export/UNION_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_offsetS_whiteBg \
# -m $IKARUS/project/dynamic-head-avatars/code/gaussian-splatting/output/Union10EMOEXP_${SUBJECT}_eval_xyz-1e0_poseLr1e-5_exprLr1e-3_600k_densify10k-300k \
# --port $GS_PORT \
# --eval --white_background --bind_to_mesh \
# --lambda_xyz 1e0 \
# --iterations 600_000 --position_lr_max_steps 600_000 --densify_from_iter 10_000 --densify_until_iter 300_000
# "


#------- run -------#
COMMAND="$HOME/usr/bin/isbatch.sh $JOB_NAME $COMMAND"

# echo $COMMAND
$COMMAND

sleep 5
GS_PORT=$((GS_PORT+1))
done



# python train.py \
# -s $IKARUS/project/dynamic-head-avatars/code/multi-view-head-tracker/export/UNION_055_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_offsetS_whiteBg \
# -m $IKARUS/project/dynamic-head-avatars/code/gaussian-splatting/output/Union10EMOEXP_055_train_xyz-1e0_poseLr1e-5_exprLr1e-3_300k_densify5k-150k \
# --port 60000 \
# --eval --white_background --bind_to_mesh \
# --lambda_xyz 1e0 \
# --iterations 300_000 --position_lr_max_steps 300_000 --densify_from_iter 5_000 --densify_until_iter 150_000
