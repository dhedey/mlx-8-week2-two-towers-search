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
import argparse
import math
from common import TrainingHyperparameters, select_device
from models import PooledTwoTowerModelHyperparameters, PooledTwoTowerModel
from trainer import ModelTrainer

if __name__ == "__main__":
    device = select_device()

    parser = argparse.ArgumentParser(description='Continue training a dual encoder model for text search')
    parser.add_argument(
        '--model',
        type=str,
        default="fixed-boosted-word2vec-linear",
    )
    parser.add_argument(
        '--end-epoch',
        type=int,
        default=None,
    )
    parser.add_argument(
        '--immediate-validation',
        type=bool,
        default=False,
    )
    args = parser.parse_args()

    model_name = args.model
    override_to_epoch = args.end_epoch

    match model_name:
        case "fixed-boosted-word2vec-linear":
            continuation = PooledTwoTowerModel.load_to_continue_training(model_name=model_name, device=device)
        case "learned-boosted-word2vec-linear":
            continuation = PooledTwoTowerModel.load_to_continue_training(model_name=model_name, device=device)
        case "learned-boosted-mini-lm-linear":
            continuation = PooledTwoTowerModel.load_to_continue_training(model_name=model_name, device=device)
        case _:
            raise ValueError(f"Unknown model name: {model_name}")

    trainer = ModelTrainer(
        model=continuation["model"].to(device),
        start_epoch=continuation["epoch"] + 1,
        start_optimizer_state=continuation["optimizer_state"],
        override_to_epoch=override_to_epoch,
        immediate_validation=args.immediate_validation,
    )
    trainer.train()



        
