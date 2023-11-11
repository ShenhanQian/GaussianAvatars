#!/bin/bash

# Define a list of subjects and sequences
# SUBJECTS=("074" "104" "218" "238" "253" "264" "302" "304" "306")  #9
# SUBJECTS=("074" "104" "140" "165" "210" "218" "238" "253" "264" "302" "304" "306")  #V2 (#12)
# SUBJECTS=("074" "104" "140" "165" "210" "218" "238" "253" "264" "302" "304" "306" "460" "494")  #V2 (#14)
# SUBJECTS=("304" "306")  # tmp
# SUBJECTS=("306")  # tmp
# SUBJECTS=("304")  # tmp


# Loop through subjects and sequences and modify the command accordingly
GS_PORT=60032

for SUBJECT in "${SUBJECTS[@]}"; do

# full model
#====================================#
# JOB_NAME="gs_${SUBJECT}_${GS_PORT}"
# COMMAND="python train.py \
# -s $IKARUS/project/dynamic-head-avatars/code/multi-view-head-tracker/export/UNION_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
# -m $IKARUS/project/dynamic-head-avatars/code/gaussian-splatting/output/Union10EMOEXP_${SUBJECT}_eval_600k_densifyTilEnd_maskBelowLine \
# --port $GS_PORT \
# --eval --white_background --bind_to_mesh \
# --position_lr_init 0.005 --position_lr_final 0.00005 --scaling_lr 0.017 \
# --lambda_xyz 1e-2 --lambda_scale 1e0 --threshold_scale 0.6 \
# --iterations 600_000 --position_lr_max_steps 600_000 --densification_interval 2_000 --opacity_reset_interval 60_000 --interval 60_000 --densify_from_iter 10_000 --densify_until_iter 600_000
# "

#====================================#
# JOB_NAME="gs_${SUBJECT}_${GS_PORT}_disableFlameStaticOffset"
# COMMAND="python train.py \
# -s $IKARUS/project/dynamic-head-avatars/code/multi-view-head-tracker/export/UNION_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
# -m $IKARUS/project/dynamic-head-avatars/code/gaussian-splatting/output/Union10EMOEXP_${SUBJECT}_eval_600k_densifyTilEnd_maskBelowLine_disableFlameStaticOffset \
# --port $GS_PORT \
# --eval --white_background --bind_to_mesh \
# --position_lr_init 0.005 --position_lr_final 0.00005 --scaling_lr 0.017 \
# --lambda_xyz 1e-2 --lambda_scale 1e0 --threshold_scale 0.6 \
# --iterations 600_000 --position_lr_max_steps 600_000 --densification_interval 2_000 --opacity_reset_interval 60_000 --interval 60_000 --densify_from_iter 10_000 --densify_until_iter 600_000 \
# --disable_flame_static_offset
# "

#====================================#
# JOB_NAME="gs_${SUBJECT}_${GS_PORT}_noDensityControl"
# COMMAND="python train.py \
# -s $IKARUS/project/dynamic-head-avatars/code/multi-view-head-tracker/export/UNION_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
# -m $IKARUS/project/dynamic-head-avatars/code/gaussian-splatting/output/Union10EMOEXP_${SUBJECT}_eval_600k_densifyTilEnd_maskBelowLine_noDensityControl \
# --port $GS_PORT \
# --eval --white_background --bind_to_mesh \
# --position_lr_init 0.005 --position_lr_final 0.00005 --scaling_lr 0.017 \
# --lambda_xyz 1e-2 --lambda_scale 1e0 --threshold_scale 0.6 \
# --iterations 600_000 --position_lr_max_steps 600_000 --densification_interval 2_000 --opacity_reset_interval 60_000 --interval 60_000 --densify_from_iter 600_000 --densify_until_iter 0
# "

#====================================#
# JOB_NAME="gs_${SUBJECT}_${GS_PORT}_xyz0"
# COMMAND="python train.py \
# -s $IKARUS/project/dynamic-head-avatars/code/multi-view-head-tracker/export/UNION_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
# -m $IKARUS/project/dynamic-head-avatars/code/gaussian-splatting/output/Union10EMOEXP_${SUBJECT}_eval_600k_densifyTilEnd_maskBelowLine_xyz0 \
# --port $GS_PORT \
# --eval --white_background --bind_to_mesh \
# --position_lr_init 0.005 --position_lr_final 0.00005 --scaling_lr 0.017 \
# --lambda_xyz 0 --threshold_xyz 1. --lambda_scale 1e0 --threshold_scale 0.6 \
# --iterations 600_000 --position_lr_max_steps 600_000 --densification_interval 2_000 --opacity_reset_interval 60_000 --interval 60_000 --densify_from_iter 10_000 --densify_until_iter 600_000
# "

#====================================#
# JOB_NAME="gs_${SUBJECT}_${GS_PORT}_xyzNoThreshold"
# COMMAND="python train.py \
# -s $IKARUS/project/dynamic-head-avatars/code/multi-view-head-tracker/export/UNION_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
# -m $IKARUS/project/dynamic-head-avatars/code/gaussian-splatting/output/Union10EMOEXP_${SUBJECT}_eval_600k_densifyTilEnd_maskBelowLine_xyzNoThreshold \
# --port $GS_PORT \
# --eval --white_background --bind_to_mesh \
# --position_lr_init 0.005 --position_lr_final 0.00005 --scaling_lr 0.017 \
# --lambda_xyz 1e-2 --threshold_xyz 0 --lambda_scale 1e0 --threshold_scale 0.6 \
# --iterations 600_000 --position_lr_max_steps 600_000 --densification_interval 2_000 --opacity_reset_interval 60_000 --interval 60_000 --densify_from_iter 10_000 --densify_until_iter 600_000
# "

#====================================#
# JOB_NAME="gs_${SUBJECT}_${GS_PORT}_scale0"
# COMMAND="python train.py \
# -s $IKARUS/project/dynamic-head-avatars/code/multi-view-head-tracker/export/UNION_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
# -m $IKARUS/project/dynamic-head-avatars/code/gaussian-splatting/output/Union10EMOEXP_${SUBJECT}_eval_600k_densifyTilEnd_maskBelowLine_scale0 \
# --port $GS_PORT \
# --eval --white_background --bind_to_mesh \
# --position_lr_init 0.005 --position_lr_final 0.00005 --scaling_lr 0.017 \
# --lambda_xyz 1e-2 --threshold_xyz 1. --lambda_scale 0 --threshold_scale 0.6 \
# --iterations 600_000 --position_lr_max_steps 600_000 --densification_interval 2_000 --opacity_reset_interval 60_000 --interval 60_000 --densify_from_iter 10_000 --densify_until_iter 600_000
# "

#====================================#
# JOB_NAME="gs_${SUBJECT}_${GS_PORT}_scaleNoThreshold"
# COMMAND="python train.py \
# -s $IKARUS/project/dynamic-head-avatars/code/multi-view-head-tracker/export/UNION_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
# -m $IKARUS/project/dynamic-head-avatars/code/gaussian-splatting/output/Union10EMOEXP_${SUBJECT}_eval_600k_densifyTilEnd_maskBelowLine_scaleNoThreshold \
# --port $GS_PORT \
# --eval --white_background --bind_to_mesh \
# --position_lr_init 0.005 --position_lr_final 0.00005 --scaling_lr 0.017 \
# --lambda_xyz 1e-2 --threshold_xyz 1. --lambda_scale 1e0 --threshold_scale 0 \
# --iterations 600_000 --position_lr_max_steps 600_000 --densification_interval 2_000 --opacity_reset_interval 60_000 --interval 60_000 --densify_from_iter 10_000 --densify_until_iter 600_000
# "

#====================================#
# JOB_NAME="gs_${SUBJECT}_${GS_PORT}_noFinetunFlameParam"
# COMMAND="python train.py \
# -s $IKARUS/project/dynamic-head-avatars/code/multi-view-head-tracker/export/UNION_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
# -m $IKARUS/project/dynamic-head-avatars/code/gaussian-splatting/output/Union10EMOEXP_${SUBJECT}_eval_600k_densifyTilEnd_maskBelowLine_noFinetunFlameParam \
# --port $GS_PORT \
# --eval --white_background --bind_to_mesh \
# --position_lr_init 0.005 --position_lr_final 0.00005 --scaling_lr 0.017 \
# --lambda_xyz 1e-2 --lambda_scale 1e0 --threshold_scale 0.6 \
# --iterations 600_000 --position_lr_max_steps 600_000 --densification_interval 2_000 --opacity_reset_interval 60_000 --interval 60_000 --densify_from_iter 10_000 --densify_until_iter 600_000 \
# --not_finetune_flame_params
# "

#------- run -------#
COMMAND="$HOME/usr/bin/isbatch.sh $JOB_NAME $COMMAND"

# echo $COMMAND
$COMMAND

sleep 1
GS_PORT=$((GS_PORT+1))
done
