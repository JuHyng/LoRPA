export TASK_NAME=glue
export DATASET_NAME=rte
export PEFT_TYPE=lorpa

bs=8
seed=$1
epoch=20

max_seq_length=512

prune_method="magnitude"
prune_strategy="top-k"

for r in 256
do
for prune_steps_ratio in "0.2_0.8"
do
for prune_ratio in "0.0_0.96875"
do
for lr in 1e-4 2e-4 3e-4
do
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
        --output_dir checkpoints/$PEFT_TYPE/$TASK_NAME/$DATASET_NAME-roberta-large/$prune_method/$prune_strategy/r_$r/$prune_steps_ratio/$prune_ratio/ \
        --overwrite_output_dir \
        --seed $seed \
        --save_strategy no \
        --evaluation_strategy epoch \
        --peft_type $PEFT_TYPE \
        --r $r \
        --initial_lora_alpha $r \
        --pruned_lora_alpha 16 \
        --label_names labels \
        --metric_for_best_model accuracy \
        --warmup_ratio 0.06 \
        --prune_steps_ratio $prune_steps_ratio \
        --prune_ratio $prune_ratio \
        --prune_strategy $prune_strategy \
        --prune_method $prune_method \
        --weight_decay 0.1
done
done
done
done