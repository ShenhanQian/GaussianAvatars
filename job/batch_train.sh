#!/bin/bash

# Define a list of subjects and sequences
# SUBJECTS=("055" "074" "104" "140" "165" "210" "218" "221" "238" "251" "253" "264" "302" "304" "306" "375")  #16
# SUBJECTS=("074" "104" "140" "165" "218" "238" "253" "264" "302" "304" "306")  #11
# SUBJECTS=("074" "104" "218" "238" "253" "264" "302" "304" "306")  #9
# SUBJECTS=("302")  # tmp


# Loop through subjects and sequences and modify the command accordingly
GS_PORT=60001

for SUBJECT in "${SUBJECTS[@]}"; do

#====================================#
# JOB_NAME="gs_${SUBJECT}_${GS_PORT}"
# COMMAND="python train.py \
# -s $IKARUS/project/dynamic-head-avatars/code/multi-view-head-tracker/export/UNION_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_offsetS_whiteBg \
# -m $IKARUS/project/dynamic-head-avatars/code/gaussian-splatting/output/Union10EMOEXP_${SUBJECT}_eval_xyz-1e0_poseLr1e-5_exprLr1e-3_600k_densify10k-300k \
# --port $GS_PORT \
# --eval --white_background --bind_to_mesh \
# --lambda_xyz 1e0 \
# --iterations 600_000 --position_lr_max_steps 600_000 --densify_from_iter 10_000 --densify_until_iter 300_000
# "


#====================================#
# JOB_NAME="gs_${SUBJECT}_${GS_PORT}"
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

sleep 1
GS_PORT=$((GS_PORT+1))
done
