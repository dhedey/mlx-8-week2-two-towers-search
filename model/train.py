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
from common import TrainingHyperparameters, select_device
from models import PooledTwoTowerModelHyperparameters, PooledTwoTowerModel
from trainer import ModelTrainer

if __name__ == "__main__":
    device = select_device()

    training_parameters = TrainingHyperparameters(
        batch_size=128,
        epochs=20,
        learning_rate=0.002,
        freeze_embeddings=True,
        dropout=0.3,
        initial_token_embeddings_kind="word2vec-boosted",
        margin=0.4,
    )

    model_parameters = PooledTwoTowerModelHyperparameters(
        comparison_embedding_size=64,
        query_tower_hidden_dimensions=[],
        doc_tower_hidden_dimensions=[],
        include_layer_norms=True,
        tokenizer="week1-word2vec",
    )

    model = PooledTwoTowerModel(
        training_parameters=training_parameters,
        model_parameters=model_parameters,
    ).to(device)

    trainer = ModelTrainer(
        model=model,
        training_parameters=training_parameters,
    )
    trainer.train()



        
