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
from models import DualEncoderModel
from trainer import ModelTrainer

if __name__ == "__main__":
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

    model, training_state = DualEncoderModel.load_for_training(args.model)

    trainer = ModelTrainer(
        model=model,
        continuation=training_state,
        override_to_epoch=args.end_epoch,
        immediate_validation=args.immediate_validation,
    )
    trainer.train()



        
