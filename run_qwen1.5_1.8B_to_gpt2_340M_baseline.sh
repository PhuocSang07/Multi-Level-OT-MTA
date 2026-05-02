#!/bin/bash
# Distill VoCuc/Qwen1.5_1.8B_SFT_Dolly (teacher, 2048-dim, 24 layers)
#        → openai-community/gpt2-medium (student, 1024-dim, 24 layers)
# Variant: BASELINE (OT distillation only, no span loss)

GPUS=(0 1)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export TOKENIZERS_PARALLELISM=false

MASTER_ADDR=localhost
MASTER_PORT=66$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1   # single-process model parallelism

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OPTS=""
OPTS+=" --model_name openai-community/gpt2-medium"
OPTS+=" --dataset.file $SCRIPT_DIR/llm_distillation/datasets/loader/dolly.py"
OPTS+=" --lr 5e-4"
OPTS+=" --num_epochs 10"
OPTS+=" --batch_size_training 4"
OPTS+=" --gradient_accumulation_steps 2"
OPTS+=" --val_batch_size 8"
OPTS+=" --output_dir $SCRIPT_DIR/output/qwen1.5-1.8B-to-gpt2-340M/baseline"
OPTS+=" --distillation"
OPTS+=" --distillation_config_model_name VoCuc/Qwen1.5_1.8B_SFT_Dolly"
OPTS+=" --distillation_config_distil_factor 0.15"
OPTS+=" --distillation_config_cross_entropy_factor 1.0"
OPTS+=" --distillation_config_student_temperature 1.0"
OPTS+=" --distillation_config_teacher_temperature 2.0"
OPTS+=" --distillation_config_pure_bf16"
OPTS+=" --student_device cuda:0"
OPTS+=" --teacher_device cuda:1"
OPTS+=" --save_step 2000"
OPTS+=" --f 1"
OPTS+=" --context_length 1024"

export NCCL_DEBUG=""
export WANDB_DISABLED=False
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=$SCRIPT_DIR

CMD="torchrun ${DISTRIBUTED_ARGS} $SCRIPT_DIR/finetuning.py ${OPTS} $@"
echo ${CMD}
mkdir -p $SCRIPT_DIR/output/qwen1.5-1.8B-to-gpt2-340M/baseline
${CMD}
