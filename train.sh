# if runing encoder-decoder, MODEL_ARCHITECTURE='encoder-decoder'
export MODEL_ARCHITECTURE='encoder'

for task in mrpc
do
for peft_type in lorpa
do
    bash run_scripts/${peft_type}/run_${task}_roberta-base.sh 42 # seed
done
done