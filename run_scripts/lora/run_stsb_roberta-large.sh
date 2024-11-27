export TASK_NAME=glue
export DATASET_NAME=stsb
export PEFT_TYPE=lora

export LARGE_MNLI_TIMESTAMP_8="seed_63/20240816-054543/"
export LARGE_MNLI_TIMESTAMP_128="seed_63/20240728-050804/"
export LARGE_MNLI_TIMESTAMP_256="/seed_63/20240725-141453/"

seed=$1

bs=8
epoch=30

max_seq_length=512

for r in 8
do
for lr in 2e-4
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
        --metric_for_best_model pearson \
        --warmup_ratio 0.06 \
        --weight_decay 0.1 \
        # --adapter_path checkpoints/lora/$TASK_NAME/mnli-roberta-large/r_$r/$LARGE_MNLI_TIMESTAMP_256
done
done