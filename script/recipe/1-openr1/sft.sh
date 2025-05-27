export CUDA_VISIBLE_DEVICES=1,2,3,4

#0.定义输出路径
OUTPUT_DIR=./output/openr1_sft
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
num_processes=$(expr length "$CUDA_VISIBLE_DEVICES" / 2 + 1)


accelerate launch --config_file=script/accelerate_configs/zero2.yaml \
            --num_processes=$num_processes easyalign/algorithms/sft.py \
            --model_name_or_path /data/public/model/Qwen2.5-1.5B-Instruct \
            --dataset_name /data/gwy/datasets/Bespoke-Stratos-17k \
            --torch_dtype "bfloat16" \
            --attn_implementation flash_attention_2 \
            --use_peft False \
            --bf16 True \
            --load_in_8bit False \
            --load_in_4bit False \
            --num_train_epochs 1 \
            --max_seq_length 4096 \
            --per_device_train_batch_size 2 \
            --gradient_accumulation_steps 8 \
            --gradient_checkpointing True \
            --learning_rate 2.0e-5 \
            --optim "adamw_torch" \
            --lr_scheduler_type "cosine" \
            --warmup_ratio 0.1 \
            --logging_steps 100 \
            --save_strategy "epoch" \
            --report_to "wandb" \
            --wandb_project "openr1" \
            --output_dir $OUTPUT_DIR 



























