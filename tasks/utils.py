from tasks.glue.dataset import task_to_keys as glue_tasks
from tasks.superglue.dataset import task_to_keys as superglue_tasks

GLUE_DATASETS = list(glue_tasks.keys())
SUPERGLUE_DATASETS = list(superglue_tasks.keys())
NER_DATASETS = ["conll2003", "conll2004", "ontonotes"]
SRL_DATASETS = ["conll2005", "conll2012"]
QA_DATASETS = ["squad", "squad_v2"]
SUM_DATASETS = ["cnn_dm", "xsum"]
NLG_DATASETS = ['e2e_nlg']


TASKS = ["glue", "superglue", "ner", "srl", "qa", "sum"]

DATASETS = GLUE_DATASETS + SUPERGLUE_DATASETS + NER_DATASETS + SRL_DATASETS + QA_DATASETS + SUM_DATASETS + NLG_DATASETS

ADD_PREFIX_SPACE = {
    'bert': False,
    'roberta': True,
    'deberta': True,
    'gpt2': True,
    'deberta-v2': True,
}

USE_FAST = {
    'bert': True,
    'roberta': True,
    'deberta': True,
    'gpt2': True,
    'deberta-v2': False,
}