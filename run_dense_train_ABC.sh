#!/bin/bash

# Define variables
FILEPATH="/home/ray/suniRet/data/sup_random_fact.txt"
OUTPUT_DIR="/home/ray/suniRet/train_output/sup_random_train_gs_gpt"
MODEL_NAME="/home/ray/suniRet/train_output/train_mixture/checkpoint-1800"
# MODEL_NAME="/home/ray/suniRet/train_output/train_f_gpt_short/checkpoint-100"
POOLING_MODE="cls"
EVAL_STEPS=100
SAVE_STEPS=10
SAVE_TOTAL_LIMIT=10
LOGGING_STEPS=10
RUN_NAME="test"
SAMPLE_TYPE="ABC"
LOSS_NAME="MNR"
FP16=false
NUM_TRAIN_EPOCHS=2
BATCH_SIZE=16
LEARNING_RATE=5e-7
WARMUP_RATIO=0.1
gradient_accumulation_steps=2
lr_scheduler_type=linear
weight_decay=0.01

# Run the Python script with arguments
python dense_trainer.py \
    --filepath "$FILEPATH" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --pooling_mode "$POOLING_MODE" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --warmup_ratio "$WARMUP_RATIO" \
    --eval_steps "$EVAL_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --logging_steps "$LOGGING_STEPS" \
    --run_name "$RUN_NAME" \
    --sample_type "$SAMPLE_TYPE" \
    --loss_name "$LOSS_NAME" \
    --fp16 "$FP16" \
    --gradient_accumulation_steps $gradient_accumulation_steps\
    --lr_scheduler_type $lr_scheduler_type \
    --weight_decay $weight_decay