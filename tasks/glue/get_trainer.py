import logging
import os
import random
import sys

from transformers import (
    AutoConfig,
    AutoTokenizer,
)

from model.utils import get_model, TaskType
from tasks.glue.dataset import GlueDataset
from training.trainer_base import BaseTrainer

import torch

from peft import PeftModel, PeftConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


logger = logging.getLogger(__name__)

def get_trainer(args, model=None, best_metrics=None):
    model_args, data_args, training_args, _= args

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )

    dataset = GlueDataset(tokenizer, data_args, training_args)
    

    if not dataset.is_regression:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=dataset.num_labels,
            label2id=dataset.label2id,
            id2label=dataset.id2label,
            finetuning_task=data_args.dataset_name,
            revision=model_args.model_revision,
        )
    else:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=dataset.num_labels,
            finetuning_task=data_args.dataset_name,
            revision=model_args.model_revision,
        )
        
    
    if model is None:
        model = get_model(model_args, TaskType.SEQUENCE_CLASSIFICATION, config)
    else:
        model.peft_config['default'].lora_alpha=model_args.pruned_lora_alpha

    # Initialize our Trainer
    trainer = BaseTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        compute_metrics=dataset.compute_metrics,
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
        best_metrics=best_metrics
    )

    return trainer, None