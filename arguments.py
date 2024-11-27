from enum import Enum
import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments, Seq2SeqTrainingArguments, GenerationConfig

from tasks.utils import *

from dataclasses import dataclass, field
from typing import Literal, Optional, Union

import torch

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.training_args
    """

    task_name: str = field(
        metadata={
            "help": "The name of the task to train on: " + ", ".join(TASKS),
            "choices": TASKS
        },
    )
    dataset_name: str = field(
        metadata={
            "help": "The name of the dataset to use: " + ", ".join(DATASETS),
            "choices": DATASETS
        }
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=True, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default=None, 
        metadata={"help": "A csv or a json file containing the test data."}
    )
    template_id: Optional[int] = field(
        default=0,
        metadata={
            "help": "The specific prompt string to use"
        }
    )
    max_source_length: Optional[int] = field(
        default=464,
        metadata={
            "help": (
                "The maximum total sequence length for source text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    
    max_target_length: Optional[int] = field(
        default=80,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    # hidden_dropout_prob: float = field(
    #     default=0.1,
    #     metadata={
    #         "help": "The dropout probability used in the models"
    #     }
    # )
    peft_type: str = field(
        default="lora",
        metadata={
            "help": "The type of PEFT to use",
            "choices": ["lora", "rslora", "dora", 'lorpa']
        }
    )
    lora: bool = field(
        default=False,
        metadata={
            "help": "Will use LoRA during training"
        }
    )
    
    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    initial_lora_alpha: int = field(default=8, metadata={"help": "Lora alpha of initial training stage"})
    pruned_lora_alpha: int = field(default=8, metadata={"help": "Lora alpha after pruning"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    init_lora_weights: bool =  field(
        default=True,
        metadata={
            "help": (
                "How to initialize the weights of the LoRA layers. Passing True (default) results in the default "
                "initialization from the reference implementation from Microsoft. Passing 'gaussian' results "
                "in Gaussian initialization scaled by the LoRA rank for linear and layers. Setting the initialization "
                "to False leads to completely random initialization and is discouraged."
                "Pass `'loftq'` to use LoftQ initialization"
            ),
        },
    )
    use_rslora: bool = field(
        default=False,
        metadata={
            "help": (
                "When set to True, uses Rank-Stabilized LoRA doi.org/10.48550/arXiv.2312.03732"
                " which sets the adapter scaling factor to `lora_alpha/math.sqrt(r)`, since it"
                " was proven to work better. Otherwise, it will use the original default"
                " value of `lora_alpha/r`."
            )
        },
    )        
    use_dora: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable 'Weight-Decomposed Low-Rank Adaptation' (DoRA). This technique decomposes the updates of the "
                "weights into two parts, magnitude and direction. Direction is handled by normal LoRA, whereas the "
                "magnitude is handled by a separate learnable parameter. This can improve the performance of LoRA, "
                "especially at low ranks. Right now, DoRA only supports linear and Conv2D layers. DoRA introduces a bigger"
                "overhead than pure LoRA, so it is recommended to merge weights for inference. For more information, "
                "see  https://arxiv.org/abs/2402.09353."
            )
        },
    )
    prune_steps_ratio: str = field(
        default="1.0",
        metadata={
            "help": "The ratio of the weights to prune. 10%->30%->30% = 0.1_0.3_0.3"
        },
    )
    prune_ratio: str = field(
        default="0.0",
        metadata={
            "help": "The ratio of the r to prune. 0%->40%->50% = 0.0_0.4_0.5"
        },
    )
    
    prune_method: str = field(
        default="none",
        metadata={
            "help": "[magnitude, random, loss]"
        },
    )
    prune_strategy: str = field(
        default="grid",
        metadata={
            "help": "method for pruning [grid, top-k]"
        },
    )
    save_model: bool = field(
        default=False,
        metadata={
            "help": "Save the model after training"
        }
    )
    
    adapter_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the adapter to load"
        }
    )
    
    

@dataclass
class QuestionAnwseringArguments:
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        },
    )
    
@dataclass
class GenerationArguments:
    """
    Arguments pertaining to generation configuration for Seq2Seq models.
    """
    max_length: Optional[int] = field(
        default=128,
        metadata={"help": "Maximum length of the generated text."}
    )
    max_new_tokens: Optional[int] = field(
        default=64,
        metadata={"help": "Maximum number of new tokens to add to the input text."}
    )
    num_beams: Optional[int] = field(
        default=4,
        metadata={"help": "Number of beams to use in beam search."}
    )
    do_sample: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use sampling during generation."}
    )
    top_k: Optional[int] = field(
        default=50,
        metadata={"help": "The number of highest probability vocabulary tokens to keep for top-k-filtering."}
    )
    top_p: Optional[float] = field(
        default=0.9,
        metadata={"help": "The cumulative probability for top-p filtering."}
    )
    temperature: Optional[float] = field(
        default=1.0,
        metadata={"help": "The value used to module the next token probabilities."}
    )
    length_penalty: Optional[float] = field(
        default=1.0,
        metadata={"help": "The exponential penalty to the length."}
    )
    no_repeat_ngram_size: Optional[int] = field(
        default=0,
        metadata={"help": "The size of ngrams that should not be repeated in the generated text."}
    )


def get_args(architecture: Optional[str] = 'encoder'):
    """Parse all the args."""
    if architecture == 'encoder':
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, QuestionAnwseringArguments))
    elif architecture == 'encoder-decoder':
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, GenerationArguments))
    else:
        raise ValueError("Architecture must be either 'encoder' or 'encoder-decoder'")

    args = parser.parse_args_into_dataclasses()

    return args