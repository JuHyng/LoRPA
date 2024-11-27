import logging
import os
import random
import sys

from transformers import (
    AutoConfig,
    AutoTokenizer,
    GPT2LMHeadModel,
)

from model.utils import TaskType
from tasks.nlg.dataset import NLGDataset
from training.trainer_base import BaseTrainer

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)

def get_trainer(args, model=None, best_metrics=None):
    model_args, data_args, training_args, _ = args

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )

    dataset = NLGDataset(tokenizer, data_args, training_args)

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        finetuning_task=data_args.dataset_name,
        revision=model_args.model_revision,
    )

    if model is None:
        model = GPT2LMHeadModel.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision,
        )
    else:
        model.peft_config['default'].lora_alpha = model_args.pruned_lora_alpha

    # Initialize our Trainer
    trainer = BaseTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
        best_metrics=best_metrics,
        compute_metrics=dataset.compute_metrics,
        preprocess_logits_for_metrics=dataset.preprocess_logits_for_metrics,
    )

    return trainer, None
