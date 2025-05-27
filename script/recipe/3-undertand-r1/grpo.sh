export CUDA_VISIBLE_DEVICES=3,4,5

#0.定义输出路径
OUTPUT_DIR=./output/undertand-r1
current_time=$(date "+%Y%m%d%H%M%S")  # 格式化当前时间为年月日时分秒
OUTPUT_DIR="${OUTPUT_DIR}_data${current_time}"
mkdir -p "$OUTPUT_DIR"
#1.定义Python文件的启动路径
SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
True_DIR="$(dirname "${ROOT_DIR}")"
True_DIR="$(dirname "${True_DIR}")"
export PYTHONPATH="${True_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
#2.将本启动脚本进行备份
SCRIPT_NAME=$(basename "$0")
DESTINATION_PATH="$OUTPUT_DIR/$SCRIPT_NAME"
cp "$0" "$DESTINATION_PATH"
#3.计算当前可见的cuda_device
num_processes=$(expr length "$CUDA_VISIBLE_DEVICES" / 2 + 1 - 1)


screen -S test1 nohup accelerate launch --config_file=script/accelerate_configs/zero2.yaml \
            --num_processes=$num_processes easyalign/algorithm_trl/grpo.py \
            --model_name_or_path /data3/public/model/Qwen2.5-Math-1.5B \
            --dataset_name /data4/gwy/R1/understand-r1-zero-main/datasets/train/math_12k/train \
            --torch_dtype "bfloat16" \
            --attn_implementation flash_attention_2 \
            --use_peft False \
            --bf16 True \
            --load_in_8bit False \
            --load_in_4bit False \
            --do_eval False \
            --num_train_epochs 1 \
            --max_prompt_length 512 \
            --max_completion_length 1024 \
            --per_device_train_batch_size 36\
            --gradient_accumulation_steps 3 \
            --num_generations 8 \
            --gradient_checkpointing True \
            --learning_rate 2e-5 \
            --optim "adamw_torch" \
            --lr_scheduler_type "cosine" \
            --warmup_ratio 0.01 \
            --logging_steps 1 \
            --log_completions True \
            --save_strategy "steps" \
            --save_steps 300 \
            --beta 0.01 \
            --template 'r1' \
            --loss_type 'grpo' \
            --reward_funcs "understand_reward"\
            --reward_weights 1.0\
            --use_vllm true \
            --vllm_device auto \
            --vllm_gpu_memory_utilization 0.7 \
            --report_to "wandb" \
            --wandb_project "openr1_grpo" \
            --output_dir $OUTPUT_DIR 



























