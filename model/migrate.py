from dataclasses import dataclass, field
import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import re
import os
import statistics
import transformers
import random
import pandas as pd
import math
from typing import Optional, Self
from common import PersistableModel
from models import PooledOneTowerModelHyperparameters, PooledTwoTowerModelHyperparameters, RNNTowerModelHyperparameters

models = [
    {
        "name": "fixed-boosted-word2vec-pooled",
        "class_name": "PooledTwoTowerModel",
    },
    {
        "name": "learned-boosted-gemma-3-1b-pooled",
        "class_name": "PooledTwoTowerModel",
    },
    {
        "name": "learned-boosted-mini-lm-pooled",
        "class_name": "PooledTwoTowerModel",
    },
    {
        "name": "fixed-boosted-word2vec-rnn",
        "class_name": "RNNTwoTowerModel",
    },
    {
        "name": "learned-boosted-word2vec-rnn",
        "class_name": "RNNTwoTowerModel",
    },
]

# OLD STATE
# torch.save({
#     "model": model.state_dict(),
#     "training_parameters": model.training_parameters.to_dict(),
#     "model_parameters": model.model_hyperparameters(),
#     "validation_metrics": model.validation_metrics,
#     "optimizer_state": optimizer.state_dict(),
#     "epoch": epoch,
# }, location)

for model in models:
    model_name = model["name"]
    class_name = model["class_name"]
    model_path = PersistableModel._model_path(model_name)
    torch.serialization.add_safe_globals([
        PooledOneTowerModelHyperparameters,
        PooledTwoTowerModelHyperparameters,
        RNNTowerModelHyperparameters
    ])

    data = torch.load(model_path, map_location='cpu')  # Load the model to ensure it exists
    if "training_parameters" in data:
        torch.save({
            "model": {
                "class_name": class_name,
                "weights": data["model"],
                "creation_state": {
                    "model_name": model_name,
                    "hyper_parameters": data["model_parameters"].to_dict(),
                    "training_parameters": data["training_parameters"],
                    "validation_metrics": data["validation_metrics"],
                }
            },
            "training": {
                "epoch": data["epoch"],
                "optimizer_state": data["optimizer_state"],
                "latest_training_loss": None,  # Old state does not have these fields
                "latest_validation_loss": None,
            },
        }, model_path)
        print("Migration complete for model:", model_name)
