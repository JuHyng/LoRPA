import logging
import os
import random
import sys

from transformers import (
    AutoConfig,
    AutoTokenizer,
    GenerationConfig
)

from model.utils import get_model, TaskType
from tasks.sum.dataset import SummarizationDataset
from training.trainer_sum import SummerizationTrainer

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)

def get_trainer(args, model=None, best_metrics=None):
    model_args, data_args, training_args, gen_args= args
    
    # Set the generation configuration
    generation_config = GenerationConfig(
        max_length=gen_args.max_length,
        max_new_tokens=gen_args.max_new_tokens,
        num_beams=gen_args.num_beams,
        do_sample=gen_args.do_sample,
        length_penalty=gen_args.length_penalty,
        no_repeat_ngram_size=gen_args.no_repeat_ngram_size,
    )

    training_args.generation_config = generation_config

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )

    dataset = SummarizationDataset(tokenizer, data_args, training_args)

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        finetuning_task=data_args.dataset_name,
        revision=model_args.model_revision,
    )

    if model is None:
        model = get_model(model_args, TaskType.SUMMARIZATION, config)
    else:
        model.peft_config['default'].lora_alpha = model_args.pruned_lora_alpha

    # Initialize our Trainer
    trainer = SummerizationTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        predict_dataset=dataset.predict_dataset if training_args.do_predict else None,
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
        best_metrics=best_metrics,
        compute_metrics=dataset.compute_metrics,
        preprocess_logits_for_metrics=dataset.preprocess_logits_for_metrics,
    )

    return trainer, None
