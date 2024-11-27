export TASK_NAME=glue
export DATASET_NAME=mnli
export PEFT_TYPE=lora

seed=$1

bs=16
lr=5e-4
epoch=30

max_seq_length=512

# bs = 16 일때 128 256에서 학습 제대로 안됨
# bs = 32, lr=7e-4로 128에서 해보기로

for r in 8
do
    timestamp=$(date "+%Y%m%d-%H%M%S")

    python run.py \
        --model_name_or_path roberta-base \
        --task_name $TASK_NAME \
        --dataset_name $DATASET_NAME \
        --do_train \
        --do_eval \
        --max_seq_length $max_seq_length \
        --per_device_train_batch_size $bs \
        --learning_rate $lr \
        --num_train_epochs $epoch \
        --output_dir checkpoints/$PEFT_TYPE/$TASK_NAME/$DATASET_NAME-roberta-base/r_$r/seed_$seed/$timestamp/ \
        --overwrite_output_dir \
        --seed $seed \
        --save_strategy no \
        --evaluation_strategy epoch \
        --peft_type $PEFT_TYPE \
        --r $r \
        --lora_alpha $r \
        --label_names labels \
        --metric_for_best_model accuracy \
        --warmup_ratio 0.06 \
        --weight_decay 0.1 \
        --save_model True 
    done