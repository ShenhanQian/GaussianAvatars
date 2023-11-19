#!/bin/bash

# Define a list of subjects and sequences
# SUBJECTS=("074" "104" "218" "238" "253" "264" "302" "304" "306")  #9
# SUBJECTS=("074" "104" "140" "165" "210" "218" "238" "253" "264" "302" "304" "306")  #V2 (#12)
# SUBJECTS=("074" "104" "140" "165" "175" "210" "218" "238" "253" "264" "302" "304" "306" "460")  #V2 (#13)


# Loop through subjects and sequences and modify the command accordingly
GS_PORT=60020

for SUBJECT in "${SUBJECTS[@]}"; do

src_folder=$IKARUS/project/dynamic-head-avatars/code/multi-view-head-tracker/export/UNION_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine


#================= localScale-localXyz (best) ===================#
# JOB_NAME="gs_${SUBJECT}_${GS_PORT}_localScale1e0-thresh0.6_localXyz1e-2-thresh1"
# COMMAND="python train.py \
# -s $src_folder \
# -m $IKARUS/project/dynamic-head-avatars/code/gaussian-splatting/output/Union10EMOEXP_${SUBJECT}_eval_600k_localScale1e0-thresh0.6_localXyz1e-2-thresh1 \
# --port $GS_PORT \
# --eval --white_background --bind_to_mesh \
# --position_lr_init 0.005 --position_lr_final 0.00005 --scaling_lr 0.017 \
# --lambda_xyz 1e-2 --lambda_scale 1e0 --threshold_scale 0.6 \
# --iterations 600_000 --position_lr_max_steps 600_000 --densification_interval 2_000 --opacity_reset_interval 60_000 --interval 60_000 --densify_from_iter 10_000 --densify_until_iter 600_000
# "

#================= metricScale-localXyz ===================#
# JOB_NAME="gs_${SUBJECT}_${GS_PORT}_metricScale5e3-thresh3e-3_localXyz1e-2-thresh1"
# COMMAND="python train.py \
# -s $src_folder \
# -m $IKARUS/project/dynamic-head-avatars/code/gaussian-splatting/output/Union10EMOEXP_${SUBJECT}_eval_600k_metricScale5e3-thresh3e-3_localXyz1e-2-thresh1 \
# --port $GS_PORT \
# --eval --white_background --bind_to_mesh \
# --position_lr_init 0.005 --position_lr_final 0.00005 --scaling_lr 0.017 \
# --lambda_xyz 1e-2 --threshold_xyz 1. \
# --metric_scale --lambda_scale 5e3 --threshold_scale 3e-3 \
# --iterations 600_000 --position_lr_max_steps 600_000 --densification_interval 2_000 --opacity_reset_interval 60_000 --interval 60_000 --densify_from_iter 10_000 --densify_until_iter 600_000
# "


#------- run -------#
COMMAND="$HOME/usr/bin/isbatch.sh $JOB_NAME $COMMAND"

# echo $COMMAND
$COMMAND

sleep 1
GS_PORT=$((GS_PORT+1))
done
