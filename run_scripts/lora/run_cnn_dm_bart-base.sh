export TASK_NAME=sum
export DATASET_NAME=cnn_dm
export PEFT_TYPE=lora

seed=$1

bs=64
epoch=25

for r in 16
do
for lr in 2e-4
do
    timestamp=$(date "+%Y%m%d-%H%M%S")

    python run.py \
        --model_name_or_path facebook/bart-base \
        --task_name $TASK_NAME \
        --dataset_name $DATASET_NAME \
        --do_train \
        --do_eval \
        --per_device_train_batch_size $bs \
        --per_device_eval_batch_size 8 \
        --learning_rate $lr \
        --num_train_epochs $epoch \
        --output_dir checkpoints/$PEFT_TYPE/$TASK_NAME/$DATASET_NAME-bart-base/r_$r/seed_$seed/$timestamp/ \
        --overwrite_output_dir \
        --pad_to_max_length \
        --seed $seed \
        --save_strategy steps \
        --evaluation_strategy steps \
        --peft_type $PEFT_TYPE \
        --r $r \
        --lora_alpha $((r*2)) \
        --label_names labels \
        --metric_for_best_model loss \
        --max_seq_length 1024 \
        --max_source_length 1024 \
        --max_target_length 128 \
        --generation_max_length 128 \
        --gradient_accumulation_steps 4 \
        --warmup_steps 500 \
        --lora_dropout 0.05 \
        --weight_decay 0.01 \
        --fp16 \
        --predict_with_generate False \
        --load_best_model_at_end \
        --save_total_limit 1 \
        --save_model
done
done