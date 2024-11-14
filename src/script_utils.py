import re
import torch
import math
from collections import defaultdict
import os
import argparse
import json

try:
    import wandb
except ImportError as e:
    pass

# Taken from https://sumit-ghosh.com/articles/parsing-dictionary-key-value-pairs-kwargs-argparse-python/
class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value_str = value.split("=")
            if value_str.replace("-", "").isnumeric():
                processed_val = int(value_str)
            elif value_str.replace("-", "").replace(".", "").isnumeric():
                processed_val = float(value_str)
            elif value_str in ["True", "true"]:
                processed_val = True
            elif value_str in ["False", "false"]:
                processed_val = False
            else:
                processed_val = value_str
            getattr(namespace, self.dest)[key] = processed_val

def summary(network):
    """Print model summary."""
    print("")
    print("Model Summary")
    print("---------------------------------------------------------------")
    for name, param in network.named_parameters():
        print(name, param.numel())
    print(
        "Total parameters:",
        sum(p.numel() for p in network.parameters() if p.requires_grad),
    )
    print("---------------------------------------------------------------")
    print("")

def aggregate_dicts(dicts):
    result = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            result[k].append(v)
    return {k: sum(v) / len(v) for k, v in result.items()}

def initialize_wandb(config):
    if config.wandb_api_key_path is not None:
        with open(config.wandb_api_key_path, "r") as f:
            os.environ["WANDB_API_KEY"] = f.read().strip()

    wandb.init(**config.wandb_kwargs)
    wandb.config.update(config)

def load_checkpoint_pretrained_encoder(checkpoint_path, model, device="cpu"):
    state = torch.load(checkpoint_path, map_location=torch.device(device))
    if "encoder_state_dict" in state:
        state_dict = state["encoder_state_dict"]
    else:
        state_dict = state["state_dict"]

    current_model_dict = model.state_dict()
    loose_state_dict = {
        k: v if v.size() == current_model_dict[k].size() else current_model_dict[k]
        for k, v in zip(current_model_dict.keys(), state_dict.values())
    }
    # Sometimes the model is saved with "backbone" prefix
    new_state_dict = {
        key.replace("backbone.", ""): value for key, value in loose_state_dict.items()
    }

    missing_keys, _ = model.load_state_dict(new_state_dict, strict=False)
    print("Missing keys when loading checkpoint: ", missing_keys)

    res = (state, model)

    return res