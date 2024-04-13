SUBJECT=304

python train.py \
	-s data/cluster/ikarus/sqian/project/dynamic-head-avatars/code/multi-view-head-tracker/export/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
	-m output/UNION10EMOEXP_${SUBJECT}_eval_600k \
	--port 60000 --eval --white_background --bind_to_mesh
