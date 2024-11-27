import logging
import os
import sys
import numpy as np
from typing import Dict

import datasets
import transformers
from transformers import set_seed, Trainer
from transformers.trainer_utils import get_last_checkpoint

from arguments import get_args

from tasks.utils import *

import torch

# def prune_r (model, r_index)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_TYPE_TO_QV = {
    "bert": ["query_encoder", "value_encoder"],
    "roberta": ["query_encoder", "value_encoder"],
    "bart": ["query_encoder", "value_encoder", "query_decoder", "value_decoder"],
}

def get_magnitude(r, lora_A, lora_B):
    
    return torch.sum(torch.abs(lora_B.weight.data[:,r] * lora_A.weight.data[r, :]))

def get_lora_from_model(model, layer, query_or_value):
    # if model is BERT
    if model.config.model_type == "bert":
        # query
        if query_or_value == "query_encoder":
            lora_A = model.bert.encoder.layer[layer].attention.self.query.lora_A.default
            lora_B = model.bert.encoder.layer[layer].attention.self.query.lora_B.default
        # value
        elif query_or_value == "value_encoder":
            lora_A = model.bert.encoder.layer[layer].attention.self.value.lora_A.default
            lora_B = model.bert.encoder.layer[layer].attention.self.value.lora_B.default
        
    # if model is RoBERTa
    elif model.config.model_type == "roberta":
        #query
        if query_or_value == "query_encoder":
            lora_A = model.roberta.encoder.layer[layer].attention.self.query.lora_A.default
            lora_B = model.roberta.encoder.layer[layer].attention.self.query.lora_B.default
        # value
        elif query_or_value == "value_encoder":
            lora_A = model.roberta.encoder.layer[layer].attention.self.value.lora_A.default
            lora_B = model.roberta.encoder.layer[layer].attention.self.value.lora_B.default
    
    elif model.config.model_type == "bart":
        if query_or_value == "query_encoder":
            lora_A = model.model.model.encoder.layers[layer].self_attn.q_proj.lora_A.default
            lora_B = model.model.model.encoder.layers[layer].self_attn.q_proj.lora_B.default
        elif query_or_value == "value_encoder":
            lora_A = model.model.model.encoder.layers[layer].self_attn.v_proj.lora_A.default
            lora_B = model.model.model.encoder.layers[layer].self_attn.v_proj.lora_B.default
        elif query_or_value == "query_decoder":
            lora_A = model.model.model.decoder.layers[layer].self_attn.q_proj.lora_A.default
            lora_B = model.model.model.decoder.layers[layer].self_attn.q_proj.lora_B.default
        elif query_or_value == "value_decoder":
            lora_A = model.model.model.decoder.layers[layer].self_attn.v_proj.lora_A.default
            lora_B = model.model.model.decoder.layers[layer].self_attn.v_proj.lora_B.default
        
    return lora_A, lora_B

def prune_rs(prune_indices, lora_A, lora_B, size):

    prune_indices = torch.tensor(prune_indices, dtype=torch.long).to(device)
    
    new_size = size[0] - len(prune_indices)

    all_indices = torch.arange(size[0])
    keep_indices = torch.tensor([i for i in all_indices if i not in prune_indices], dtype=torch.long).to(device)

    new_lora_A_data = torch.index_select(lora_A.weight.data, 0, keep_indices)
    new_lora_B_data = torch.index_select(lora_B.weight.data, 1, keep_indices)
    
    lora_A.out_features = new_size
    lora_B.in_features = new_size
    
    lora_A.weight.data = new_lora_A_data
    lora_B.weight.data = new_lora_B_data

    return keep_indices


def prune (model, r_size, num_prune, pruned_dict, prune_strategy, prune_method):
    
    for layer in range(model.config.num_hidden_layers):
        
        if layer not in pruned_dict:
            pruned_dict[layer] = {}
        
        for query_or_value in MODEL_TYPE_TO_QV[model.config.model_type]:
            if query_or_value not in pruned_dict[layer]:
                pruned_dict[layer][query_or_value] = [0 for i in range(r_size)]
            
            magnitudes = torch.zeros(r_size).to(device)
            loss = torch.zeros(r_size).to(device)
        
            lora_A, lora_B = get_lora_from_model(model, layer, query_or_value)
            
            size = lora_A.weight.size()
            
            if prune_method == "magnitude":
                for r in range(r_size):
                    magnitudes[r]+=get_magnitude(r, lora_A, lora_B)
            
            if prune_method == "magnitude":
                if prune_strategy == "top-k":    
                    prune_indices = torch.argsort(magnitudes)[:num_prune].to(device)
            elif prune_method == "random":
                prune_indices = torch.randperm(r_size)[:num_prune].to(device)
                
                
            keep_indices = prune_rs(prune_indices, lora_A, lora_B, size)
            pruned_dict[layer][query_or_value] = keep_indices
    
    r_size-=num_prune
            
    return model, r_size, pruned_dict
    

   