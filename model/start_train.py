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
from models import PooledTwoTowerModelHyperparameters, PooledTwoTowerModel, RNNTwoTowerModel, RNNTowerModelHyperparameters
from trainer import ModelTrainer

if __name__ == "__main__":
    device = select_device()

    parser = argparse.ArgumentParser(description='Train a dual encoder model for text search')
    parser.add_argument(
        '--model',
        type=str,
        default="fixed-boosted-word2vec-pooled",
    )
    args = parser.parse_args()

    model_name = args.model

    match model_name:
        case "fixed-boosted-word2vec-pooled":
            training_parameters = TrainingHyperparameters(
                batch_size=128,
                epochs=20,
                learning_rate=0.002,
                dropout=0.3,
                margin=0.4,
                initial_token_embeddings_kind="default",
                freeze_embeddings=False,
                initial_token_embeddings_boost_kind="sqrt-inverse-frequency",
                freeze_embedding_boosts=True,
            )

            model_parameters = PooledTwoTowerModelHyperparameters(
                comparison_embedding_size=64,
                query_tower_hidden_dimensions=[],
                doc_tower_hidden_dimensions=[],
                include_layer_norms=True,
                tokenizer="week1-word2vec",
            )

            model = PooledTwoTowerModel(
                model_name=model_name,
                training_parameters=training_parameters,
                model_parameters=model_parameters,
            )
        case "learned-boosted-word2vec-pooled":
            training_parameters = TrainingHyperparameters(
                batch_size=128,
                epochs=20,
                learning_rate=0.002,
                dropout=0.3,
                margin=0.4,
                initial_token_embeddings_kind="default",
                freeze_embeddings=True,
                initial_token_embeddings_boost_kind="ones",
                freeze_embedding_boosts=False, # Learn boosts
            )
            model_parameters = PooledTwoTowerModelHyperparameters(
                comparison_embedding_size=64,
                query_tower_hidden_dimensions=[],
                doc_tower_hidden_dimensions=[],
                include_layer_norms=True,
                tokenizer="week1-word2vec",
            )
            model = PooledTwoTowerModel(
                model_name=model_name,
                training_parameters=training_parameters,
                model_parameters=model_parameters,
            )
        case "learned-boosted-mini-lm-pooled":
            training_parameters = TrainingHyperparameters(
                batch_size=128,
                epochs=20,
                learning_rate=0.002,
                dropout=0.3,
                margin=0.4,
                initial_token_embeddings_kind="default",
                freeze_embeddings=True,
                initial_token_embeddings_boost_kind="ones",
                freeze_embedding_boosts=False, # Learn boosts
            )

            model_parameters = PooledTwoTowerModelHyperparameters(
                comparison_embedding_size=64,
                query_tower_hidden_dimensions=[],
                doc_tower_hidden_dimensions=[],
                include_layer_norms=True,
                tokenizer="pretrained:sentence-transformers/all-MiniLM-L6-v2",
            )

            model = PooledTwoTowerModel(
                model_name=model_name,
                training_parameters=training_parameters,
                model_parameters=model_parameters,
            )
        case "fixed-boosted-word2vec-rnn":
            training_parameters = TrainingHyperparameters(
                batch_size=128,
                epochs=20,
                learning_rate=0.002,
                dropout=0.3,
                margin=0.4,
                initial_token_embeddings_kind="default",
                freeze_embeddings=False,
                initial_token_embeddings_boost_kind="sqrt-inverse-frequency",
                freeze_embedding_boosts=True,
            )

            model_parameters = RNNTowerModelHyperparameters(
                comparison_embedding_size=64,
                query_tower_hidden_dimensions=[128, 64],
                doc_tower_hidden_dimensions=[128, 64],
                include_layer_norms=True,
                tokenizer="week1-word2vec",
            )

            model = RNNTwoTowerModel(
                model_name=model_name,
                training_parameters=training_parameters,
                model_parameters=model_parameters,
            )
        case "learned-boosted-word2vec-rnn":
            training_parameters = TrainingHyperparameters(
                batch_size=128,
                epochs=20,
                learning_rate=0.002,
                dropout=0.3,
                margin=0.4,
                initial_token_embeddings_kind="default",
                freeze_embeddings=True,
                initial_token_embeddings_boost_kind="ones",
                freeze_embedding_boosts=False, # Learn boosts
            )
            model_parameters = RNNTowerModelHyperparameters(
                comparison_embedding_size=64,
                query_tower_hidden_dimensions=[128, 64],
                doc_tower_hidden_dimensions=[128, 64],
                include_layer_norms=True,
                tokenizer="week1-word2vec",
            )
            model = RNNTwoTowerModel(
                model_name=model_name,
                training_parameters=training_parameters,
                model_parameters=model_parameters,
            )
        case "learned-boosted-mini-lm-rnn":
            training_parameters = TrainingHyperparameters(
                batch_size=128,
                epochs=20,
                learning_rate=0.002,
                dropout=0.3,
                margin=0.4,
                initial_token_embeddings_kind="default",
                freeze_embeddings=True,
                initial_token_embeddings_boost_kind="ones",
                freeze_embedding_boosts=False, # Learn boosts
            )
            model_parameters = RNNTowerModelHyperparameters(
                comparison_embedding_size=64,
                query_tower_hidden_dimensions=[128, 64],
                doc_tower_hidden_dimensions=[128, 64],
                include_layer_norms=True,
                tokenizer="pretrained:sentence-transformers/all-MiniLM-L6-v2",
            )
            model = RNNTwoTowerModel(
                model_name=model_name,
                training_parameters=training_parameters,
                model_parameters=model_parameters,
            )
        case _:
            raise ValueError(f"Unknown model name: {model_name}")

    trainer = ModelTrainer(model=model.to(device), validate_and_save_after_epochs=1)
    trainer.train()



        
