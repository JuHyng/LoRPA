import logging
import os
import sys
import numpy as np
from typing import Dict

import datasets
from model.utils import TaskType, get_model
from tasks.glue.dataset import GlueDataset
from tasks.superglue.dataset import SuperGlueDataset
import transformers
from transformers import set_seed
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForSequenceClassification

from arguments import get_args

from tasks.utils import *

import torch
import wandb

from prune import *

from peft import PeftModel, PeftConfig

os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)

def train(trainer, resume_from_checkpoint=None, last_checkpoint=None, save_model=False):
    checkpoint = None
    if resume_from_checkpoint is not None:
        checkpoint = resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    if save_model:
        trainer.save_model()
    # trainer.save_model()

    metrics = train_result.metrics
    
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    trainer.log_best_metrics()

def evaluate(trainer):
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

def predict(trainer, predict_dataset=None):
    if predict_dataset is None:
        logger.info("No dataset is available for testing")

    elif isinstance(predict_dataset, dict):
        
        for dataset_name, d in predict_dataset.items():
            logger.info("*** Predict: %s ***" % dataset_name)
            predictions, labels, metrics = trainer.predict(d, metric_key_prefix="predict")
            predictions = np.argmax(predictions, axis=2)

            trainer.log_metrics("predict", metrics)
            trainer.save_metrics("predict", metrics)

    else:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        predictions = np.argmax(predictions, axis=2)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

if __name__ == '__main__':
    model_architecture=os.environ.get("MODEL_ARCHITECTURE")
    args = get_args(model_architecture)

    model_args, data_args, training_args, _= args

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    

    if not os.path.isdir("checkpoints") or not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")

    if data_args.task_name.lower() == "superglue":
        assert data_args.dataset_name.lower() in SUPERGLUE_DATASETS
        from tasks.superglue.get_trainer import get_trainer

    elif data_args.task_name.lower() == "glue":
        assert data_args.dataset_name.lower() in GLUE_DATASETS
        from tasks.glue.get_trainer import get_trainer
        
    elif data_args.task_name.lower() == "sum":
        assert data_args.dataset_name.lower() in SUM_DATASETS
        from tasks.sum.get_trainer import get_trainer
    else:
        raise NotImplementedError('Task {} is not implemented. Please choose a task from: {}'.format(data_args.task_name, ", ".join(TASKS)))
    
    
    
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)
    
    trainer, predict_dataset = get_trainer(args)

    if training_args.do_train:
        if model_args.peft_type == "lorpa":  
            # 1. compute number of entire steps
            
            model_args, data_args, training_args, qa_args = args
                
            r_size = model_args.r
            batch_size = training_args.per_device_train_batch_size
            num_epochs = training_args.num_train_epochs
            
            wandb_run_name=f"{model_args.prune_method}_{model_args.model_name_or_path}_{data_args.dataset_name}_start_r:{model_args.r}"
            wandb.init(project="LoRPA", name=wandb_run_name, config=model_args, reinit=True)
            
            grouping = [model_args.peft_type, 
                        model_args.model_name_or_path, 
                        model_args.prune_strategy, 
                        data_args.dataset_name,
                        data_args.task_name,
                        model_args.prune_method, 
                        str(model_args.r),
                        model_args.prune_ratio,
                        model_args.prune_steps_ratio]
            
            for group in grouping:
                wandb.init(group=group)
                
            print({"Total training epochs": num_epochs})
            
            prune_steps_ratio = [float(ratio) for ratio in model_args.prune_steps_ratio.split("_")]
            prune_ratio = [float(ratio) for ratio in model_args.prune_ratio.split("_")]
            prune_r_size_list = [int(r_size * each_prune_ratio) for each_prune_ratio in prune_ratio]
            print({"Prune r size list": prune_r_size_list})
            
            prune_epochs = prune_steps_ratio.copy()
            prune_epochs = [int(num_epochs * ratio) for ratio in prune_steps_ratio]
            print({"Prune epochs": prune_epochs})
            print({"Prune ratio": prune_ratio})
            
            full_ft_epochs = prune_epochs[0]
            print(f"Fine-tuning all r for epochs: {full_ft_epochs}")
            model_args.lora_alpha = model_args.initial_lora_alpha
            
            # Fine-tuning all r
            training_args.num_train_epochs = full_ft_epochs
            args = (model_args, data_args, training_args, qa_args)
            trainer, predict_dataset = get_trainer(args)
            
            
            train(trainer, training_args.resume_from_checkpoint, last_checkpoint)
            wandb.finish()
            
            # # Pruning & training
            max_prune_epoch = full_ft_epochs
            
            #create pruned r list (list with length r_size)
            pruned_dict = {}
            prune_strategy = model_args.prune_strategy
            prune_method = model_args.prune_method
            
            for i in range(1, len(prune_epochs)):
                max_prune_epoch = prune_epochs[i]
                
                prev_best_metrics=trainer.best_metrics
                
                model = trainer.model
                # prune model
                model, pruned_r_size, pruned_dict = prune(model=model, 
                      r_size=r_size, 
                      num_prune=prune_r_size_list[i],
                      pruned_dict=pruned_dict, 
                      prune_strategy=prune_strategy, 
                      prune_method=prune_method)
                
                r_size=pruned_r_size
                
                wandb_run_name=f"{model_args.prune_method}_{model_args.model_name_or_path}_{data_args.dataset_name}_r:{r_size}_epoch:~{max_prune_epoch}"
                wandb.init(project="LoRPA", name=wandb_run_name, config=model_args)
                for group in grouping:
                    wandb.init(group=group)
                    
                print({"Pruned r size": pruned_r_size})
                print({"Pruned dict": pruned_dict})
                print({"model":model})
                
                training_args.num_train_epochs = max_prune_epoch
                model_args.lora_alpha = model_args.pruned_lora_alpha
                args = (model_args, data_args, training_args, qa_args)
                
                trainer, predict_dataset = get_trainer(args, model=model)
                
                # resume from checkpoint
                train(trainer)
                
                wandb.finish()
            
        else:
            wandb_run_name = (f"LoRA_{model_args.model_name_or_path}_{data_args.dataset_name}_r:{model_args.r}")
            wandb.init(project="LoRPA", name=wandb_run_name, config=model_args)
            
            grouping= [model_args.peft_type,model_args.model_name_or_path, data_args.dataset_name, data_args.task_name, str(model_args.r)]
            for group in grouping:
                wandb.init(group=group)
            train(trainer, training_args.resume_from_checkpoint, last_checkpoint, save_model=model_args.save_model)
        
                        
                
            
   