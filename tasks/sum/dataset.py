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

class SummarizationDataset():
    def __init__(self, tokenizer: AutoTokenizer, data_args, training_args) -> None:
        super().__init__()
        if data_args.dataset_name == "cnn_dm":
            raw_datasets = load_dataset("cnn_dailymail", "3.0.0")
            self.sentence1_key = "article"
        elif data_args.dataset_name == "xsum":
            raw_datasets = load_dataset("xsum")
            self.sentence1_key = "document"
        else:
            raise ValueError("Unsupported dataset. Please use either 'cnn_dm' or 'xsum'.")

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
                
        self.metric_rouge = evaluate.load("rouge")
        self.metric_bleu = evaluate.load("bleu")
        # self.metric_bertscore = evaluate.load("bertscore")
        self.data_collator = DataCollatorForSeq2Seq(tokenizer, padding=self.padding, max_length=self.max_seq_length)

    def preprocess_function(self, examples):
        # Tokenize the texts
        args = (
            (examples[self.sentence1_key],)
        )
        result = self.tokenizer(*args, padding=self.padding, max_length=self.max_source_length, truncation=True)
        # input(f"Tokenized Input Example: {result['input_ids'][:5]}")  # 일부 샘플 로그로 확인
        #다시 디코딩해서 확인해보기
        # input(f"Decoded Input Example: {self.tokenizer.decode(result['input_ids'][0])}")
        
        # Adding labels for summarization tasks
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(examples["highlights"] if self.data_args.dataset_name == "cnn_dm" else examples["summary"],
                                    max_length=self.max_target_length,
                                    padding=self.padding,
                                    truncation=True)
        result["labels"] = labels["input_ids"]
        # input(f"Tokenized Label Example: {result['labels'][:5]}")  # 일부 샘플 로그로 확인
        #다시 디코딩해서 확인해보기
        # input(f"Decoded Label Example: {self.tokenizer.decode(result['labels'][0])}")
        
        
        return result
    
    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions       
        label_ids = p.label_ids
        
        if isinstance(preds, tuple):
            preds = preds[0]

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
        label_ids = np.where(label_ids != -100, label_ids, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        with open("decoded_preds.txt", "w") as f:
            f.write("\n".join(decoded_preds))
        with open("decoded_labels.txt", "w") as f:
            f.write("\n".join(decoded_labels))
            
        
        # Make sure references are in the expected format for BLEU and BERTScore
        formatted_labels = [[label] for label in decoded_labels]
        
        # Compute ROUGE score
        rouge_result = self.metric_rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        
        # Compute BLEU score
        bleu_result = self.metric_bleu.compute(predictions=decoded_preds,
                                               references=formatted_labels)

        # Compute BERTScore
        # bertscore_result = self.metric_bertscore.compute(predictions=decoded_preds,
        #                                                  references=decoded_labels,
        #                                                  lang="en")
        
        # Merge all metrics
        result = {
            **rouge_result,
            "bleu": bleu_result["bleu"] * 100,  # Scale BLEU to percentage
            # "bertscore_precision": np.mean(bertscore_result["precision"]) * 100,
            # "bertscore_recall": np.mean(bertscore_result["recall"]) * 100,
            # "bertscore_f1": np.mean(bertscore_result["f1"]) * 100,
        }

        return result
    
    def preprocess_logits_for_metrics(self, logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids, labels
