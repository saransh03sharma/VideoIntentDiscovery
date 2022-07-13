#!/bin/bash

OUTPUT_ROOT=./out/unimodal/both
SCRIPT=./scripts/run_unimodal_ner.py

if [ "$1" = "--multimodal" ]; then
  echo "Using multi-modal model"
  SCRIPT=./scripts/run_multimodal_ner.py
  OUTPUT_ROOT=./out/multimodal/both
fi

GPUID=$2
echo "Run on GPU $GPUID"

# data
# PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..
PROJECT_ROOT=.
DATA_ROOT=$PROJECT_ROOT/data/intent/semi/

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
#	for TRAIN_BATCH in 32 16
    for TRAIN_BATCH in 16
	do
		for GRAD_ACC_STEPS in 1 2
		do
			# output
			TOTAL_BATCH_SIZE=$((TRAIN_BATCH*GRAD_ACC_STEPS))
			OUTPUT=$OUTPUT_ROOT/semi/${MODEL_NAME}_${TOTAL_BATCH_SIZE}_${LR}/

	#		[ -e $OUTPUT/script  ] || mkdir -p $OUTPUT/script
	#		cp -f $(readlink -f "$0") $OUTPUT/script
	#		rsync -ruzC --exclude-from=$PROJECT_ROOT/.gitignore --exclude 'dataset' --exclude 'pretrained_model' --exclude 'outputs' $PROJECT_ROOT/ $OUTPUT/src

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
			  --max_seq_length 128 \
			  --overwrite_output_dir \
			  --eval_all_checkpoints
		done
		  
	done
done
