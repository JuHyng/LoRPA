export TASK_NAME=glue
export DATASET_NAME=cola
export PEFT_TYPE=lora

seed=$1

bs=4
lr=6e-5
epoch=20

max_seq_length=128

for r in 256
do
for lr in 2e-5 3e-5 4e-5
do
    timestamp=$(date "+%Y%m%d-%H%M%S")

    python run.py \
        --model_name_or_path roberta-large \
        --task_name $TASK_NAME \
        --dataset_name $DATASET_NAME \
        --do_train \
        --do_eval \
        --max_seq_length $max_seq_length \
        --per_device_train_batch_size $bs \
        --learning_rate $lr \
        --num_train_epochs $epoch \
        --output_dir checkpoints/$PEFT_TYPE/$TASK_NAME/$DATASET_NAME-roberta-large/r_$r/seed_$seed/$timestamp/ \
        --overwrite_output_dir \
        --seed $seed \
        --save_strategy no \
        --evaluation_strategy epoch \
        --peft_type $PEFT_TYPE \
        --r $r \
        --lora_alpha $((r*2)) \
        --label_names labels \
        --metric_for_best_model matthews_correlation \
        --warmup_ratio 0.06 \
        --weight_decay 0.1
done
done