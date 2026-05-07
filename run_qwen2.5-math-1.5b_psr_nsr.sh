#!/usr/bin/env bash

# Single GPU execution as requested
export CUDA_VISIBLE_DEVICES=0

math_train_path=./data/math/train.parquet
model_name=Qwen/Qwen2.5-Math-1.5B
output_dir=./checkpoints/trl-qwen2.5-math-1.5b-psr-nsr

# Hyperparameters
lr=1e-6
batch_size=128
mini_batch_size=2
epochs=10
max_response_length=512

echo "Starting TRL-based Confidence-Weighted NSR training..."
echo "Model: $model_name"

python train_trl.py \
    --model_name "$model_name" \
    --train_files "$math_train_path" \
    --output_dir "$output_dir" \
    --learning_rate "$lr" \
    --batch_size "$batch_size" \
    --mini_batch_size "$mini_batch_size" \
    --epochs "$epochs" \
    --max_response_length "$max_response_length" \
    "$@"
