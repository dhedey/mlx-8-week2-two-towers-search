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
from common import TrainingHyperparameters, ModelLoader, select_device
from tokenizer import Word2VecTokenizer, TokenizerBase

def prepare_tokens_for_embedding_bag(tokens_list: list[list[int]], device):
    """
    See https://docs.pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html
    """
    flattened_tokens = []
    offsets = []
    current_offset = 0

    for tokens in tokens_list:
        offsets.append(current_offset)
        flattened_tokens.extend(tokens)
        current_offset += len(tokens)

    return [
        torch.tensor(flattened_tokens, dtype=torch.long).to(device),
        torch.tensor(offsets, dtype=torch.long).to(device),
    ]
    
class PooledTowerModel(nn.Module):
    def __init__(
            self,
            default_token_embeddings: torch.Tensor,
            training_parameters: TrainingHyperparameters,
            hidden_layer_sizes: list[int],
            include_layer_norms: bool,
            output_size: int,
        ):
        super(PooledTowerModel, self).__init__()
        self.average_pooling = nn.EmbeddingBag.from_pretrained(
            embeddings=default_token_embeddings,
            freeze=training_parameters.freeze_embeddings,
            mode='mean',
        )
        dimension_sizes = hidden_layer_sizes

        input_sizes = [default_token_embeddings.shape[1]] + dimension_sizes
        output_sizes = dimension_sizes + [output_size]
        hidden_layer_sizes = zip(input_sizes[:-1], output_sizes[:-1])
        self.hidden_layers = nn.Sequential(*[
            HiddenLayer(input_size, output_size, include_layer_norms, training_parameters.dropout) for input_size, output_size in hidden_layer_sizes
        ])

        self.output_layer = nn.Linear(input_sizes[-1], output_sizes[-1])


    def forward(self, tokens: list[list[int]]):
        device = next(self.parameters()).device
        flattened_tokens, offsets = prepare_tokens_for_embedding_bag(tokens, device)

        x = self.average_pooling(flattened_tokens, offsets)
        x = self.hidden_layers(x)
        x = self.output_layer(x)

        return x

class HiddenLayer(nn.Module):
    def __init__(self, input_size, output_size, include_layer_norm, dropout):
        super(HiddenLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)
        if include_layer_norm:
            self.layer_norm = nn.LayerNorm(output_size)
        else:
            self.layer_norm = nn.Identity()

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.layer_norm(x)
        return x

class DualEncoderModel(nn.Module):
    """A base class for all our dual encoder models."""
    def __init__(self):
        super(DualEncoderModel, self).__init__()

    def get_device(self):
        return next(self.parameters()).device

    def tokenize_query(self, query: str) -> list[int]:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def tokenize_document(self, document: str) -> list[int]:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def embed_tokenized_queries(self, tokenized_queries: list[list[int]]):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def embed_tokenized_documents(self, tokenized_documents: list[list[int]]):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def model_hyperparameters(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

@dataclass
class PooledTwoTowerModelHyperparameters:
    tokenizer: str
    comparison_embedding_size: int
    query_tower_hidden_dimensions: list[int]
    doc_tower_hidden_dimensions: list[int]
    include_layer_norms: bool

    def to_dict(self):
        return vars(self)

class PooledTwoTowerModel(DualEncoderModel):
    tokenizer: TokenizerBase

    def __init__(self, training_parameters: TrainingHyperparameters, model_parameters: PooledTwoTowerModelHyperparameters):
        super(PooledTwoTowerModel, self).__init__()

        match model_parameters.tokenizer:
            case "week1-word2vec":
                tokenizer = Word2VecTokenizer.load()
                default_token_embeddings = tokenizer.generate_default_embeddings(training_parameters.initial_token_embeddings_kind)
            case _:
                raise ValueError(f"Unknown tokenizer: {model_parameters.tokenizer}")
            
        self.tokenizer = tokenizer

        self.query_tower=PooledTowerModel(
            training_parameters=training_parameters,
            hidden_layer_sizes=model_parameters.query_tower_hidden_dimensions,
            output_size=model_parameters.comparison_embedding_size,
            include_layer_norms=model_parameters.include_layer_norms,
            default_token_embeddings=default_token_embeddings,
        )
        self.document_tower=PooledTowerModel(
            training_parameters=training_parameters,
            hidden_layer_sizes=model_parameters.doc_tower_hidden_dimensions,
            output_size=model_parameters.comparison_embedding_size,
            include_layer_norms=model_parameters.include_layer_norms,
            default_token_embeddings=default_token_embeddings,
        )
        self._model_hyperparameters = model_parameters

    @classmethod
    def load_for_evaluation(cls, model_name: str, device):
        model_loader = ModelLoader()
        loaded_model_data = model_loader.load_model_data(
            model_name=model_name,
            model_parameters_class=PooledTwoTowerModelHyperparameters,
            device=device,
        )
        model = cls(
            training_parameters=TrainingHyperparameters.for_prediction(),
            model_parameters=loaded_model_data["model_parameters"],
        ).to(device)
        model.load_state_dict(loaded_model_data["model"])
        model.eval()

        return model

    def tokenize_query(self, query: str) -> list[int]:
        return self.tokenizer.tokenize(query)
    
    def tokenize_document(self, document: str) -> list[int]:
        return self.tokenizer.tokenize(document)

    def embed_tokenized_queries(self, tokenized_queries: list[list[int]]):
        return self.query_tower(tokenized_queries)

    def embed_tokenized_documents(self, tokenized_documents: list[list[int]]):
        return self.document_tower(tokenized_documents)
    
    def model_hyperparameters(self):
        return self._model_hyperparameters

@dataclass
class PooledOneTowerModelHyperparameters:
    tokenizer: str
    comparison_embedding_size: int
    hidden_dimensions: list[int]
    include_layer_norms: bool

    def to_dict(self):
        return vars(self)

class PooledOneTowerModel(DualEncoderModel):
    tokenizer: TokenizerBase

    def __init__(self, training_parameters: TrainingHyperparameters, model_parameters: PooledOneTowerModelHyperparameters):
        super(PooledOneTowerModel, self).__init__()

        match model_parameters.tokenizer:
            case "week1-word2vec":
                tokenizer = Word2VecTokenizer.load()
                default_token_embeddings = tokenizer.generate_default_embeddings(training_parameters.initial_token_embeddings_kind)
            case _:
                raise ValueError(f"Unknown tokenizer: {model_parameters.tokenizer}")
            
        self.tokenizer = tokenizer

        self.tower=PooledTowerModel(
            training_parameters=training_parameters,
            hidden_layer_sizes=model_parameters.hidden_dimensions,
            output_size=model_parameters.comparison_embedding_size,
            include_layer_norms=model_parameters.include_layer_norms,
            default_token_embeddings=default_token_embeddings,
        )
        self._model_hyperparameters = model_parameters

    @classmethod
    def load_for_evaluation(cls, model_name: str, device):
        model_loader = ModelLoader()
        loaded_model_data = model_loader.load_model_data(
            model_name=model_name,
            model_parameters_class=PooledOneTowerModelHyperparameters,
            device=device,
        )
        model = cls(
            training_parameters=TrainingHyperparameters.for_prediction(),
            model_parameters=loaded_model_data["model_parameters"],
        ).to(device)
        model.load_state_dict(loaded_model_data["model"])
        model.eval()

        return model

    def tokenize_query(self, query: str) -> list[int]:
        return self.tokenizer.tokenize(query)
    
    def tokenize_document(self, document: str) -> list[int]:
        return self.tokenizer.tokenize(document)

    def embed_tokenized_queries(self, tokenized_queries: list[list[int]]):
        return self.tower(tokenized_queries)

    def embed_tokenized_documents(self, tokenized_documents: list[list[int]]):
        return self.tower(tokenized_documents)
    
    def model_hyperparameters(self):
        return self._model_hyperparameters

def load_model_for_evaluation(model_name: str) -> DualEncoderModel:
    device = select_device()
    match model_name:
        case "two-tower-boosted-word2vec-linear":
            return PooledTwoTowerModel.load_for_evaluation(
                model_name=model_name,
                device=device,
            )
        case "one-tower-boosted-word2vec-linear":
            return PooledOneTowerModel.load_for_evaluation(
                model_name=model_name,
                device=device,
            )
        case _:
            raise ValueError(f"Unknown model name: {model_name}")