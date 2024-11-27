export TASK_NAME=sum
export DATASET_NAME=xsum
export PEFT_TYPE=lorpa

seed=$1

bs=64
epoch=25

prune_method="magnitude"
prune_strategy="top-k"

for r in 256
do
for lr in 2e-4
do
for prune_steps_ratio in "0.2_0.8"
do
for prune_ratio in "0.0_0.9375"
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
        --prune_steps_ratio $prune_steps_ratio \
        --prune_ratio $prune_ratio \
        --prune_strategy $prune_strategy \
        --prune_method $prune_method \
        --r $r \
        --initial_lora_alpha $r \
        --pruned_lora_alpha 16 \
        --label_names labels \
        --metric_for_best_model rouge1 \
        --max_seq_length 1024 \
        --max_source_length 960 \
        --max_target_length 64 \
        --generation_max_length 64 \
        --gradient_accumulation_steps 4 \
        --warmup_steps 500 \
        --lora_dropout 0.05 \
        --weight_decay 0.01 \
        --fp16 \
        --load_best_model_at_end \
        --save_total_limit 1 \
        --save_model
done
done
done
done