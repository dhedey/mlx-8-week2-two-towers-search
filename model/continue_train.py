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
from models import DualEncoderModel
from trainer import ModelTrainer

if __name__ == "__main__":
    device = select_device()

    parser = argparse.ArgumentParser(description='Continue training a dual encoder model for text search')
    parser.add_argument(
        '--model',
        type=str,
        default="fixed-boosted-word2vec-pooled",
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

    model, training_state = DualEncoderModel.load_for_training(model_name)

    trainer = ModelTrainer(
        model=model,
        start_epoch=training_state.epoch + 1,
        start_optimizer_state=training_state.optimizer_state,
        override_to_epoch=override_to_epoch,
        immediate_validation=args.immediate_validation,
    )
    trainer.train()



        
