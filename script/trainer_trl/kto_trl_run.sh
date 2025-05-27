export CUDA_VISIBLE_DEVICES=1,2,3

#0.定义输出路径
OUTPUT_DIR=./model_output/kto
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

#1.loss_type --- kto
#KTO: Model Alignment as Prospect Theoretic Optimization

accelerate launch --main_process_port 29501 --config_file=script/accelerate_configs/zero2.yaml \
            --num_processes=$num_processes easyalign/algorithm_trl/kto.py \
            --model_name_or_path /data1/gwy/training_model/Alpaca7b_reproduce \
            --torch_dtype "bfloat16" \
            --attn_implementation eager \
            --use_peft True \
            --bf16 True \
            --load_in_8bit False \
            --load_in_4bit False \
            --do_train True \
            --num_train_epochs 1 \
            --per_device_train_batch_size 2 \
            --gradient_accumulation_steps 1 \
            --max_length 1024 \
            --max_prompt_length 512 \
            --beta 0.1 \
            --loss_type kto \
            --desirable_weight 1.0 \
            --undesirable_weight 1.0 \
            --gradient_checkpointing True \
            --learning_rate 1.41e-5 \
            --optim "adamw_torch" \
            --lr_scheduler_type "cosine" \
            --warmup_ratio 0.1 \
            --logging_steps 100 \
            --save_strategy "epoch" \
            --report_to "wandb" \
            --output_dir $OUTPUT_DIR \



    





















