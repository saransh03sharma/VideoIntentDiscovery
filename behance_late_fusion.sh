#!/bin/bash

SCRIPT=./scripts/run_multimodal_ner.py
OUTPUT_ROOT=./out/multimodal/both

FEATURE_TYPE=$1
if [ "$1" = "2d" ]; then
  echo "Using 2D ResNet features"
  FEATURE_DIR=$2
  FEAT_MAX_LEN=16
  FEAT_DIM=2048
elif [ "$1" = "3d" ]; then
    echo "Using 3D ResNeXt features"
  FEATURE_DIR=$2
  FEAT_MAX_LEN=16
  FEAT_DIM=2048
elif [ "$1" = "slowfast" ]; then
    echo "Using SlowFast features"
  FEATURE_DIR=$2
  FEAT_MAX_LEN=16
  FEAT_DIM=2304
fi

GPUID=$3
echo "Run on GPU $GPUID"

# data
# PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..
PROJECT_ROOT=.
DATA_ROOT=$PROJECT_ROOT/data/bid/
OUTPUT_ROOT=./out/multimodal/both

export PYTHONPATH=$PYTHONPATH:/ssd-playpen1/adyasha/projects/Behance/

# model
MODEL_TYPE=roberta
MODEL_NAME=roberta-base

ADAM_EPS=1e-8
ADAM_BETA1=0.9
ADAM_BETA2=0.98
WARMUP=200

TRAIN_BATCH=32
EVAL_BATCH=64

# self-training parameters
REINIT=0
BEGIN_STEP=100000000 #high BEGIN_STEP prevents self training
LABEL_MODE=soft
PERIOD=450
HP_LABEL=5.9


EPOCH=5
SEED=0
WEIGHT_DECAY=1e-4

# params
LR=1e-5


for LR in 5e-5 1e-5 1e-4
do
	for TRAIN_BATCH in 16
	do
		for GRAD_ACC_STEPS in 1 2
		do
			# output
			TOTAL_BATCH_SIZE=$((TRAIN_BATCH*GRAD_ACC_STEPS))
			OUTPUT=$OUTPUT_ROOT/late_fusion_${FEATURE_TYPE}/${MODEL_NAME}_${TOTAL_BATCH_SIZE}_${LR}/

			python $SCRIPT --data_dir $DATA_ROOT \
			  --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME \
			  --learning_rate $LR \
			  --weight_decay $WEIGHT_DECAY \
			  --adam_epsilon $ADAM_EPS \
			  --adam_beta1 $ADAM_BETA1 \
			  --adam_beta2 $ADAM_BETA2 \
			  --num_train_epochs $EPOCH \
			  --warmup_steps $WARMUP \
			  --per_gpu_train_batch_size $TRAIN_BATCH \
			  --per_gpu_eval_batch_size $EVAL_BATCH \
			  --gradient_accumulation_steps $GRAD_ACC_STEPS \
			  --logging_steps 500 \
			  --save_steps 500 \
			  --do_train \
			  --do_eval \
			  --do_predict \
			  --evaluate_during_training \
			  --output_dir $OUTPUT \
			  --cache_dir $PROJECT_ROOT/pretrained_model \
			  --seed $SEED \
			  --max_seq_length 70 \
			  --overwrite_output_dir \
			  --eval_all_checkpoints \
			  --late_fusion \
			  --feature_dir $FEATURE_DIR \
			  --feature_type $FEATURE_TYPE \
			  --video_feature_max_len $FEAT_MAX_LEN \
			  --video_embed_dim $FEAT_DIM \
			  --clip_length 5
		done
	done
done
