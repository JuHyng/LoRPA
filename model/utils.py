from enum import Enum

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForMultipleChoice,
    AutoModelForSeq2SeqLM
)

from peft import LoraConfig, get_peft_model, PeftModel

class TaskType(Enum):
    TOKEN_CLASSIFICATION = 1,
    SEQUENCE_CLASSIFICATION = 2,
    QUESTION_ANSWERING = 3,
    MULTIPLE_CHOICE = 4,
    SUMMARIZATION = 5
 
AUTO_MODELS = {
    TaskType.TOKEN_CLASSIFICATION: AutoModelForTokenClassification,
    TaskType.SEQUENCE_CLASSIFICATION: AutoModelForSequenceClassification,
    TaskType.QUESTION_ANSWERING: AutoModelForQuestionAnswering,
    TaskType.MULTIPLE_CHOICE: AutoModelForMultipleChoice,
    TaskType.SUMMARIZATION: AutoModelForSeq2SeqLM
}

def get_model(model_args, task_type: TaskType, config: AutoConfig, fix_bert: bool = False):
    if model_args.peft_type in ["lora", "dora", "rslora", "lorpa"]:
        model_class = AUTO_MODELS[task_type]
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision,
        )
        lora_config = LoraConfig(
            r=model_args.r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            init_lora_weights=model_args.init_lora_weights,
            use_rslora=model_args.use_rslora,
            use_dora=model_args.use_dora
        )
        
        if model_args.adapter_path:
            model = PeftModel(model, lora_config)
            model.load_adapter(model_args.adapter_path, "default")
            model.set_adapter("default")
        else:
            model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        
    elif model_args.peft_type is 'full_finetuning':
        model_class = AUTO_MODELS[task_type]
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision
            =model_args.model_revision,
        )
    else:
        # error
        pass
        
    
    return model