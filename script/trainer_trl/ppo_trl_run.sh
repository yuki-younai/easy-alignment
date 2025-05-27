
export CUDA_VISIBLE_DEVICES=1,2,3


#0.定义输出路径
OUTPUT_DIR=./output/ppo
current_time=$(date "+%Y%m%d%H%M%S")  # 格式化当前时间为年月日时分秒
OUTPUT_DIR="${OUTPUT_DIR}_data${current_time}"
mkdir -p "$OUTPUT_DIR"
#1.定义Python文件的启动路径
SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
True_DIR="$(dirname "${ROOT_DIR}")"
export PYTHONPATH="${True_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
#2.将本启动脚本进行备份
SCRIPT_NAME=$(basename "$0")
DESTINATION_PATH="$OUTPUT_DIR/$SCRIPT_NAME"
cp "$0" "$DESTINATION_PATH"
#3.计算当前可见的cuda_device
num_processes=$(expr length "$CUDA_VISIBLE_DEVICES" / 2 + 1)


accelerate launch --main_process_port 29501 --config_file script/accelerate_configs/zero2.yaml \
    --num_processes=$num_processes easyalign/algorithm_trl/ppo.py\
    --model_name_or_path  /data4/gwy/model/EleutherAI_pythia-1b-deduped__sft__tldr \
    --sft_model_path /data4/gwy/model/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path /data4/gwy/model/EleutherAI_pythia-1b-deduped__reward__tldr \
    --dataset_name /data4/gwy/datasets/tldr-preference-sft-trl-style \
    --torch_dtype "bfloat16" \
    --attn_implementation eager \
    --use_peft False \
    --bf16 True \
    --load_in_8bit False \
    --load_in_4bit False \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --local_rollout_forward_batch_size 16 \
    --num_ppo_epochs 4 \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --gradient_checkpointing True \
    --learning_rate 3e-6 \
    --optim "adamw_torch" \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.1 \
    --logging_steps 100 \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_strategy "epoch" \
    --report_to "wandb" \
    --output_dir $OUTPUT_DIR 