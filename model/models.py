from dataclasses import dataclass, field
import torch.nn.functional as F
import torch.nn as nn
import torch
import re
import os
import statistics
import transformers
import random
import pandas as pd
import math
from typing import Optional, Self
from .common import PersistableModel, PersistableData
from .tokenizer import get_tokenizer, TokenizerBase

@dataclass
class TrainingHyperparameters(PersistableData):
    batch_size: int
    epochs: int
    learning_rate: float
    freeze_embeddings: bool
    freeze_embedding_boosts: bool
    dropout: float
    initial_token_embeddings_kind: str
    initial_token_embeddings_boost_kind: str
    margin: float

def prepare_tokens_for_embedding_bag(tokens_list: list[list[int]], device):
    """
    See https://docs.pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html
    """
    flattened_tokens = []
    offsets = []
    mean_weights = []
    current_offset = 0

    for tokens in tokens_list:
        offsets.append(current_offset)
        flattened_tokens.extend(tokens)
        current_offset += len(tokens)

        if len(tokens) > 0:
            mean_weight = 1/len(tokens)
            mean_weights.extend(mean_weight for _ in tokens)


    return [
        torch.tensor(flattened_tokens, dtype=torch.long).to(device),
        torch.tensor(offsets, dtype=torch.long).to(device),
        torch.tensor(mean_weights, dtype=torch.float).to(device),
    ]

def prepare_tokens_for_rnn(tokens_list: list[list[int]], device):
    """
    Prepare tokens for RNN processing - keep as sequences
    """
    # Find max sequence length for padding
    max_length = max(len(tokens) for tokens in tokens_list)

    # Pad all sequences to max_length
    padded_sequences = []
    for tokens in tokens_list:
        # padding at the start as rnn recalls the last tokens better
        padded_seq =  [0] * (max_length - len(tokens)) + tokens
        padded_sequences.append(padded_seq)

    return torch.tensor(padded_sequences, dtype=torch.long).to(device)

class PooledTowerModel(nn.Module):
    def __init__(
            self,
            default_token_embeddings: torch.Tensor,
            default_token_embedding_boosts: torch.Tensor,
            training_parameters: TrainingHyperparameters,
            hidden_layer_sizes: list[int],
            include_layer_norms: bool,
            output_size: int,
        ):
        super(PooledTowerModel, self).__init__()
        self.embedding_boosts = nn.Embedding.from_pretrained(
            embeddings=default_token_embedding_boosts.unsqueeze(dim=1), # Expand to an (E, 1) shape
            freeze=training_parameters.freeze_embedding_boosts,
        )
        self.embedding_sum = nn.EmbeddingBag.from_pretrained(
            embeddings=default_token_embeddings,
            freeze=training_parameters.freeze_embeddings,
            mode='sum', # NOTE: To use the embedding boosts, we have to use `sum`, so we apply the mean later on
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
        flattened_tokens, offsets, mean_weighting = prepare_tokens_for_embedding_bag(tokens, device)
        per_token_boosts = self.embedding_boosts(flattened_tokens).squeeze(dim=1)

        x = self.embedding_sum(
            flattened_tokens,
            offsets,
            per_sample_weights=per_token_boosts * mean_weighting,
        )
        x = self.hidden_layers(x)
        x = self.output_layer(x)

        return x

class RNNTowerModel(nn.Module):
    def __init__(
            self,
            default_token_embeddings: torch.Tensor,
            training_parameters: TrainingHyperparameters,
            hidden_layer_sizes: list[int],
            include_layer_norms: bool,
            output_size: int,
        ):
        super(RNNTowerModel, self).__init__()

        self.embeddings = nn.Embedding.from_pretrained(
            default_token_embeddings,
            freeze=training_parameters.freeze_embeddings
        )

        # RNN hidden size should match the first hidden layer size
        rnn_hidden_size = hidden_layer_sizes[0] if hidden_layer_sizes else output_size

        self.rnn = nn.RNN(
            input_size=default_token_embeddings.shape[1],  # embedding dimension
            hidden_size=rnn_hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Build hidden layers starting from RNN output
        dimension_sizes = hidden_layer_sizes
        input_sizes = [rnn_hidden_size] + dimension_sizes  # Start from RNN hidden size
        output_sizes = dimension_sizes + [output_size]
        hidden_layer_sizes = zip(input_sizes[:-1], output_sizes[:-1])
        self.hidden_layers = nn.Sequential(*[
            HiddenLayer(input_size, output_size, include_layer_norms, training_parameters.dropout)
            for input_size, output_size in hidden_layer_sizes
        ])

        # Output layer takes the last hidden layer output
        self.output_layer = nn.Linear(input_sizes[-1], output_sizes[-1])

    def forward(self, sequences):
        # sequences is now a tensor of shape [batch_size, seq_len]

        # Get embeddings
        embedded_sequences = self.embeddings(sequences)  # [batch_size, seq_len, embedding_dim]

        # Process through RNN
        rnn_output, hidden = self.rnn(embedded_sequences)  # [batch_size, seq_len, hidden_size]

        # Take last hidden state for each sequence
        x = rnn_output[:, -1, :]  # [batch_size, hidden_size]

        # Pass through the same hidden layers and output layer
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

class DualEncoderModel(PersistableModel):
    validation_metrics: Optional[dict] = None

    """A base class for all our dual encoder models."""
    def __init__(self, model_name: str, training_parameters: TrainingHyperparameters, model_parameters: PersistableData):
        super(DualEncoderModel, self).__init__()
        self.model_name = model_name
        self.training_parameters = training_parameters
        self.model_parameters=model_parameters

    def build_creation_state(self) -> dict:
        return {
            "model_name": self.model_name,
            "hyper_parameters": self.model_parameters.to_dict(),
            "training_parameters": self.training_parameters.to_dict(),
            "validation_metrics": self.validation_metrics,
        }

    @classmethod
    def create(cls, creation_state: dict, for_evaluation_only: bool) -> Self:
        """This method should return a new model from the creation state."""
        model = cls(
            model_name=creation_state["model_name"],
            training_parameters=TrainingHyperparameters.from_dict(creation_state["training_parameters"]),
            model_parameters=cls.hyper_parameters_class().from_dict(creation_state["hyper_parameters"]),
        )
        model.validation_metrics = creation_state.get("validation_metrics", None)
        return model

    @classmethod
    def hyper_parameters_class(cls) -> type[PersistableData]:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def tokenize_query(self, query: str) -> list[int]:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def tokenize_document(self, document: str) -> list[int]:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def embed_tokenized_queries(self, tokenized_queries: list[list[int]]):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def embed_tokenized_documents(self, tokenized_documents: list[list[int]]):
        raise NotImplementedError("This method should be implemented by subclasses.")

class ModelLoader:
    def __init__(self):
        self.folder = os.path.dirname(__file__)

    def save_model_data(self, model: DualEncoderModel, optimizer, epoch):
        location = self.model_location(model.model_name)
        torch.save({
            "model": model.state_dict(),
            "training_parameters": model.training_parameters.to_dict(),
            "model_parameters": model.model_hyperparameters(),
            "validation_metrics": model.validation_metrics,
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
        }, location)
        print(f"Model saved to {location}")

    def load_model_data(self, model_name, model_parameters_class, device):
        model_location = self.model_location(model_name)
        loaded_data = torch.load(model_location, map_location=device)

        print(f"Loaded model {model_name}")

        return loaded_data

    def model_location(self, model_name):
        return os.path.join(self.folder, "data", f"{model_name}.pt")

@dataclass
class PooledTwoTowerModelHyperparameters(PersistableData):
    tokenizer: str
    comparison_embedding_size: int
    query_tower_hidden_dimensions: list[int]
    doc_tower_hidden_dimensions: list[int]
    include_layer_norms: bool

class PooledTwoTowerModel(DualEncoderModel):
    tokenizer: TokenizerBase

    def __init__(self, model_name: str, training_parameters: TrainingHyperparameters, model_parameters: PooledTwoTowerModelHyperparameters):
        super(PooledTwoTowerModel, self).__init__(
            model_name=model_name,
            training_parameters=training_parameters,
            model_parameters=model_parameters,
        )

        tokenizer = get_tokenizer(model_parameters.tokenizer)
        default_token_embeddings = tokenizer.generate_default_embeddings(training_parameters.initial_token_embeddings_kind)
        default_token_embedding_boosts = tokenizer.generate_default_embedding_boosts(training_parameters.initial_token_embeddings_boost_kind)

        self.tokenizer = tokenizer

        self.query_tower=PooledTowerModel(
            training_parameters=training_parameters,
            hidden_layer_sizes=model_parameters.query_tower_hidden_dimensions,
            output_size=model_parameters.comparison_embedding_size,
            include_layer_norms=model_parameters.include_layer_norms,
            default_token_embeddings=default_token_embeddings,
            default_token_embedding_boosts=default_token_embedding_boosts,
        )
        self.document_tower=PooledTowerModel(
            training_parameters=training_parameters,
            hidden_layer_sizes=model_parameters.doc_tower_hidden_dimensions,
            output_size=model_parameters.comparison_embedding_size,
            include_layer_norms=model_parameters.include_layer_norms,
            default_token_embeddings=default_token_embeddings,
            default_token_embedding_boosts=default_token_embedding_boosts,
        )

    @classmethod
    def hyper_parameters_class(cls) -> type[PersistableData]:
        return PooledTwoTowerModelHyperparameters

    def tokenize_query(self, query: str) -> list[int]:
        return self.tokenizer.tokenize(query)

    def tokenize_document(self, document: str) -> list[int]:
        return self.tokenizer.tokenize(document)

    def embed_tokenized_queries(self, tokenized_queries: list[list[int]]):
        return self.query_tower(tokenized_queries)

    def embed_tokenized_documents(self, tokenized_documents: list[list[int]]):
        return self.document_tower(tokenized_documents)

@dataclass
class RNNTowerModelHyperparameters(PersistableData):
    tokenizer: str
    comparison_embedding_size: int
    query_tower_hidden_dimensions: list[int]
    doc_tower_hidden_dimensions: list[int]
    include_layer_norms: bool

class RNNTwoTowerModel(DualEncoderModel):
    tokenizer: TokenizerBase

    def __init__(self, model_name: str, training_parameters: TrainingHyperparameters, model_parameters: RNNTowerModelHyperparameters):
        super(RNNTwoTowerModel, self).__init__(
            model_name=model_name,
            training_parameters=training_parameters,
            model_parameters=model_parameters,
        )

        tokenizer = get_tokenizer(model_parameters.tokenizer)
        default_token_embeddings = tokenizer.generate_default_embeddings(training_parameters.initial_token_embeddings_kind)

        self.tokenizer = tokenizer

        self.query_tower = RNNTowerModel(
            default_token_embeddings=default_token_embeddings,
            training_parameters=training_parameters,
            hidden_layer_sizes=model_parameters.query_tower_hidden_dimensions,
            include_layer_norms=model_parameters.include_layer_norms,
            output_size=model_parameters.comparison_embedding_size,
        )

        self.document_tower = RNNTowerModel(
            default_token_embeddings=default_token_embeddings,
            training_parameters=training_parameters,
            hidden_layer_sizes=model_parameters.doc_tower_hidden_dimensions,
            include_layer_norms=model_parameters.include_layer_norms,
            output_size=model_parameters.comparison_embedding_size,
        )

    @classmethod
    def hyper_parameters_class(cls) -> type[PersistableData]:
        return RNNTowerModelHyperparameters

    def tokenize_query(self, query: str) -> list[int]:
        return self.tokenizer.tokenize(query)

    def tokenize_document(self, document: str) -> list[int]:
        return self.tokenizer.tokenize(document)

    def embed_tokenized_queries(self, tokenized_queries: list[list[int]]):
        sequences = prepare_tokens_for_rnn(tokenized_queries, self.get_device())
        return self.query_tower(sequences)

    def embed_tokenized_documents(self, tokenized_documents: list[list[int]]):
        sequences = prepare_tokens_for_rnn(tokenized_documents, self.get_device())
        return self.document_tower(sequences)

@dataclass
class PooledOneTowerModelHyperparameters(PersistableData):
    tokenizer: str
    comparison_embedding_size: int
    hidden_dimensions: list[int]
    include_layer_norms: bool

class PooledOneTowerModel(DualEncoderModel):
    tokenizer: TokenizerBase

    def __init__(self, model_name: str, training_parameters: TrainingHyperparameters, model_parameters: PooledOneTowerModelHyperparameters):
        super(PooledOneTowerModel, self).__init__(
            model_name=model_name,
            training_parameters=training_parameters,
            model_parameters=model_parameters,
        )

        tokenizer = get_tokenizer(model_parameters.tokenizer)
        default_token_embeddings = tokenizer.generate_default_embeddings(training_parameters.initial_token_embeddings_kind)
        default_token_embedding_boosts = tokenizer.generate_default_embedding_boosts(training_parameters.initial_token_embeddings_boost_kind)

        self.tokenizer = tokenizer

        self.tower=PooledTowerModel(
            training_parameters=training_parameters,
            hidden_layer_sizes=model_parameters.hidden_dimensions,
            output_size=model_parameters.comparison_embedding_size,
            include_layer_norms=model_parameters.include_layer_norms,
            default_token_embeddings=default_token_embeddings,
            default_token_embedding_boosts=default_token_embedding_boosts,
        )

    @classmethod
    def hyper_parameters_class(cls) -> type[PersistableData]:
        return PooledOneTowerModelHyperparameters

    def tokenize_query(self, query: str) -> list[int]:
        return self.tokenizer.tokenize(query)

    def tokenize_document(self, document: str) -> list[int]:
        return self.tokenizer.tokenize(document)

    def embed_tokenized_queries(self, tokenized_queries: list[list[int]]):
        return self.tower(tokenized_queries)

    def embed_tokenized_documents(self, tokenized_documents: list[list[int]]):
        return self.tower(tokenized_documents)

if __name__ == "__main__":
    query = "What is the weather like in New York City?"

    documents = [
        "My name is John Doe and I live in New York City.",
        "I am a software engineer with a passion for machine learning.",
        "The weather in New York City is often unpredictable.",
        "I enjoy hiking and exploring new places on weekends.",
        "My favorite programming language is Python, especially for data science tasks."
    ]

    model_names = [
        "fixed-boosted-word2vec-pooled",
        "learned-boosted-mini-lm-pooled",
        "fine-tuned-mini-lm-pooled",
        "fixed-boosted-word2vec-rnn",
        "learned-boosted-word2vec-rnn",
    ]

    print(f"Showing different model results for query: {query}")
    print()

    for model_name in model_names:

        print("==========================")
        print(f"Loading model {model_name}...")
        model, training_state = DualEncoderModel.load(model_name, for_evaluation_only=True)
        model.eval()

        training_state.optimizer_state = None  # Don't print it!
        print(f"Training state: {training_state.to_dict()}")
        print(f"Validation metrics: {model.validation_metrics}")

        document_embeddings = model.embed_tokenized_documents(
            [model.tokenize_document(doc) for doc in documents]
        )

        query_embedding = model.embed_tokenized_queries(
            [model.tokenize_query(query)]
        )

        similarities = [
            {
                "document": documents[index],
                "similarity": score.item()
            }
            for (index, score) in enumerate(F.cosine_similarity(query_embedding, document_embeddings))
        ]
        ordered_results = sorted(similarities, key=lambda x: x["similarity"], reverse=True)

        print()
        print(f"Example results for query: {query}")
        print()
        for result in ordered_results:
            document = result["document"]
            similarity = result["similarity"]
            print(f"Similarity: {similarity:.3f} | Document: {document}")
        print()
