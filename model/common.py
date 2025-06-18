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

@dataclass
class TrainingHyperparameters:
    batch_size: int
    epochs: int
    learning_rate: float
    freeze_embeddings: bool
    dropout: float
    initial_token_embeddings_kind: str
    margin: float

    @classmethod
    def for_prediction(cls):
        return cls(
            batch_size=1,
            epochs=0,
            learning_rate=0,
            freeze_embeddings=True,
            dropout=0,
            margin=0,
            initial_token_embeddings_kind="zeroes",
        )

    def to_dict(self):
        return vars(self)
    
def select_device():
    DEVICE_IF_MPS_SUPPORT = 'cpu' # or 'mps' - but it doesn't work well with EmbeddingBag
    device = torch.device('cuda' if torch.cuda.is_available() else DEVICE_IF_MPS_SUPPORT if torch.backends.mps.is_available() else 'cpu')
    
    print(f'Selected device: {device}')
    return device

class ModelLoader:
    def __init__(self):
        self.folder = os.path.dirname(__file__)

    def save_model_data(self, model_name, model, model_parameters, training_parameters, optimizer, epoch, validation_metrics):
        location = self.model_location(model_name)
        torch.save({
            "model": model.state_dict(),
            "training_parameters": training_parameters.to_dict(),
            "model_parameters": model_parameters,
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "validation_metrics": validation_metrics,
        }, location)
        print(f"Model saved to {location}")

    def load_model_data(self, model_name, model_parameters_class, device):
        torch.serialization.add_safe_globals([model_parameters_class])

        model_location = self.model_location(model_name)
        loaded_data = torch.load(model_location, map_location=device)

        print(f"Loaded model {model_name}...")

        return loaded_data

    def model_location(self, model_name):
        return os.path.join(self.folder, "data", f"{model_name}.pt")