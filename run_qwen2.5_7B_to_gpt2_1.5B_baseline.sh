#!/bin/bash
# Distill VoCuc/Qwen2.5-7B-Instruct-Dolly-SFT (teacher, 3584-dim, 28 layers)
#        → openai-community/gpt2-xl (student, 1600-dim, 48 layers)
# Variant: BASELINE (OT distillation only, no span loss)

GPUS=(0 1 2 3 4 5 6 7)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export TOKENIZERS_PARALLELISM=false

MASTER_ADDR=localhost
MASTER_PORT=66$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1   # single-process model parallelism; teacher uses device_map=auto

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OPTS=""
OPTS+=" --model_name openai-community/gpt2-xl"
OPTS+=" --dataset.file $SCRIPT_DIR/llm_distillation/datasets/loader/dolly.py"
OPTS+=" --lr 1e-3"
OPTS+=" --num_epochs 10"
OPTS+=" --batch_size_training 2"
OPTS+=" --gradient_accumulation_steps 4"
OPTS+=" --val_batch_size 8"
OPTS+=" --output_dir $SCRIPT_DIR/output/qwen2.5-7B-to-gpt2-1.5B/baseline"
OPTS+=" --distillation"
OPTS+=" --distillation_config_model_name VoCuc/Qwen2.5-7B-Instruct-Dolly-SFT"
OPTS+=" --distillation_config_distil_factor 0.15"
OPTS+=" --distillation_config_cross_entropy_factor 1.0"
OPTS+=" --distillation_config_student_temperature 1.0"
OPTS+=" --distillation_config_teacher_temperature 2.0"
OPTS+=" --distillation_config_pure_bf16"
OPTS+=" --student_device cuda:0"
OPTS+=" --teacher_device auto"
OPTS+=" --save_step 2000"
OPTS+=" --f 1"
OPTS+=" --context_length 1024"
OPTS+=" --use_peft"
OPTS+=" --lora_r 256"
OPTS+=" --lora_alpha 8"
OPTS+=" --lora_dropout 0.1"

export NCCL_DEBUG=""
export WANDB_DISABLED=False
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=$SCRIPT_DIR

CMD="torchrun ${DISTRIBUTED_ARGS} $SCRIPT_DIR/finetuning.py ${OPTS} $@"
echo ${CMD}
mkdir -p $SCRIPT_DIR/output/qwen2.5-7B-to-gpt2-1.5B/baseline
${CMD}
