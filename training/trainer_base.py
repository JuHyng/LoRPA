import logging
from typing import Dict, OrderedDict
import torch

from transformers import Trainer
# from lightning.pytorch import Trainer

import wandb

logger = logging.getLogger(__name__)

_default_log_level = logging.INFO
logger.setLevel(_default_log_level)



class BaseTrainer(Trainer):
    def __init__(self, *args, predict_dataset = None, best_metrics=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.predict_dataset = predict_dataset
        if  best_metrics is not None:
            self.best_metrics = best_metrics
        else:
            self.best_metrics = OrderedDict({
                "best_epoch": 0,
                f"best_eval_{self.args.metric_for_best_model}": 0,
            })
           
    def log_best_metrics(self):
        self.log_metrics("best", self.best_metrics)
        self.save_metrics("best", self.best_metrics, combined=False)
            
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
            
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                if torch.is_tensor(grad_norm):
                    grad_norm = grad_norm.detach().cpu().item()
                logs["grad_norm"] = grad_norm
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)
            
            if metrics[f"eval_{self.args.metric_for_best_model}"] > self.best_metrics[f"best_eval_{self.args.metric_for_best_model}"]:
                self.best_metrics["best_epoch"] = epoch
                self.best_metrics[f"best_eval_{self.args.metric_for_best_model}"] = metrics[f"eval_{self.args.metric_for_best_model}"]

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                # logger.info(metric_to_check)
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])
                
            logger.info(f"***** Epoch {epoch}: Best results *****")
            for key, value in self.best_metrics.items():
                logger.info(f"{key} = {value}")
            self.log(self.best_metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
