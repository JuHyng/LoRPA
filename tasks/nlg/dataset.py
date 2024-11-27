import torch
from torch.utils import data
from torch.utils.data import Dataset
from datasets.arrow_dataset import Dataset as HFDataset
from datasets.load import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
    DataCollatorForSeq2Seq,
)
import numpy as np
import logging
import evaluate


logger = logging.getLogger(__name__)

class NLGDataset():
    def __init__(self, tokenizer: AutoTokenizer, data_args, training_args) -> None:
        super().__init__()
        if data_args.dataset_name == "e2e_nlg":
            raw_datasets = load_dataset("e2e_nlg")
            self.sentence1_key = "meaning_representation"
        else:
            raise ValueError("Unsupported dataset. Please use 'e2e_nlg'.")

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.training_args = training_args

        if data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            self.padding = False

        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
        self.max_source_length = min(data_args.max_source_length, tokenizer.model_max_length)
        self.max_target_length = data_args.max_target_length

        raw_datasets = raw_datasets.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

        if training_args.do_train:
            self.train_dataset = raw_datasets["train"]
            if data_args.max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(range(data_args.max_train_samples))

        if training_args.do_eval:
            self.eval_dataset = raw_datasets["validation"]
            if data_args.max_eval_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))

        if training_args.do_predict or data_args.dataset_name is not None or data_args.test_file is not None:
            self.predict_dataset = raw_datasets["test"]
            if data_args.max_predict_samples is not None:
                self.predict_dataset = self.predict_dataset.select(range(data_args.max_predict_samples))
                
        self.metrics = {
            "bleu": evaluate.load("bleu"),
            "nist": evaluate.load("nist"),
            "meteor": evaluate.load("meteor"),
            "rouge_l": evaluate.load("rouge"),
            "cider": evaluate.load("cider")
        }
        self.data_collator = DataCollatorForSeq2Seq(tokenizer, padding=self.padding, max_length=self.max_seq_length)

    def preprocess_function(self, examples):
        # Tokenize the texts
        args = (
            (examples[self.sentence1_key],)
        )
        result = self.tokenizer(*args, padding=self.padding, max_length=self.max_source_length, truncation=True)
        
        # Adding labels for E2E NLG tasks
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(examples["human_reference"],
                                    max_length=self.max_target_length,
                                    padding=self.padding,
                                    truncation=True)
        result["labels"] = labels["input_ids"]

        return result
    
    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions       
        label_ids = p.label_ids
        
        if isinstance(preds, tuple):
            preds = preds[0]

        # Convert predictions to CPU only when necessary to avoid GPU idle time
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()

        if isinstance(label_ids, torch.Tensor):
            label_ids = label_ids.detach().cpu().numpy()

        # Ensure that preds and label_ids have the same number of examples
        if len(preds) != len(label_ids):
            logger.error("Mismatch in number of predictions and labels.")
            logger.error(f"Predictions: {preds.shape}")
            logger.error(f"Labels: {label_ids.shape}")
            raise ValueError(f"Mismatch in number of predictions ({len(preds)}) and references ({len(label_ids)})")

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        result = {}
        for metric_name, metric in self.metrics.items():
            result[metric_name] = metric.compute(predictions=decoded_preds, references=decoded_labels)

        return result
    
    def preprocess_logits_for_metrics(self, logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids, labels