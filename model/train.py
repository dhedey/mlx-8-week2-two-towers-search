from dataclasses import dataclass, field
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import random
import re
import os

@dataclass
class TrainingHyperparameters:
    batch_size: int
    epochs: int
    learning_rate: float
    freeze_embeddings: bool
    dropout: float

    @classmethod
    def for_prediction(cls):
        return cls(
            batch_size=1,
            epochs=0,
            learning_rate=0,
            freeze_embeddings=True,
            dropout=0,
        )

    def to_dict(self):
        return vars(self)

@dataclass
class ModelHyperparameters:
    comparison_embedding_size: int = 128
    query_tower_hidden_dimensions: list[int] = field(default_factory=lambda: [128, 64])
    doc_tower_hidden_dimensions: list[int] = field(default_factory=lambda: [128, 64])
    include_layer_norms: bool = True

    def to_dict(self):
        return vars(self)

@dataclass
class ModelTokenizer:
    token_map: dict[str, int]
    default_token_embeddings: torch.Tensor

    @classmethod
    def load(cls):
        folder = os.path.dirname(__file__)
        word_vectors = torch.load(folder + '/word_vectors.pt')
        return cls(
            token_map={word: i for i, word in enumerate(word_vectors["vocabulary"])},
            default_token_embeddings=word_vectors["embeddings"],
        )
    
    def vocabulary_size(self):
        return len(self.token_map)
    
    def embedding_size(self):
        return self.default_token_embeddings.shape[1]
    
    def tokenize(self, string):
        filtered_title_words = re.sub(r'[^a-z0-9 ]', '', string.lower()).split()
        mapped_words = [self.token_map[word] for word in filtered_title_words if word in self.token_map]
        return mapped_words

def prepare_test_batch(tokenizer: ModelTokenizer, raw_batch, full_dataset, device):
    queries = []
    good_documents = []
    bad_documents = []
    dataset_len = len(full_dataset)
    document_count_for_each_query = []

    if dataset_len <= 1:
        raise ValueError("Dataset must contain more than one item to sample bad documents.")

    for query_row in raw_batch:
        queries.append(query_row["query"])
        good_passages = query_row["passages"]["passage_text"]
        good_documents.extend(good_passages)
        
        good_passage_count = len(good_passages)
        document_count_for_each_query.append(good_passage_count)

        # Sample bad documents randomly
        for i in range(good_passage_count):
            random_row = None
            while random_row is None:
                random_index = random.randint(0, len(full_dataset) - 1)
                random_row = full_dataset[random_index]
                if random_row['query_id'] == query_row['query_id']:
                    random_row = None
             
            random_passage = random.sample(random_row["passages"]["passage_text"], k=1)[0]
            bad_documents.append(random_passage)
    
    return {
        "queries": prepare_tokens_for_embedding_bag(queries, tokenizer, device),
        "good_documents": prepare_tokens_for_embedding_bag(good_documents, tokenizer, device),
        "bad_documents": prepare_tokens_for_embedding_bag(bad_documents, tokenizer, device),
        "document_count_for_each_query": document_count_for_each_query, # List not tensor
    }

def prepare_tokens_for_embedding_bag(strings, tokenizer: ModelTokenizer, device):
    """
    See https://docs.pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html
    """
    flattened_tokens = []
    offsets = []
    current_offset = 0

    for string in strings:
        offsets.append(current_offset)
        tokens = tokenizer.tokenize(string)
        flattened_tokens.extend(tokens)
        current_offset += len(tokens)

    return {
        "flattened_tokens": torch.tensor(flattened_tokens, dtype=torch.long).to(device),
        "offsets": torch.tensor(offsets, dtype=torch.long).to(device)
    }

def calculate_triplet_loss(query_vectors, good_document_vectors, bad_document_vectors, margin: float = 0.2):
    """
    Calculate the loss for a single query-document pair. 
    """
    good_similarity = F.cosine_similarity(query_vectors, good_document_vectors, dim=1)
    bad_similarity = F.cosine_similarity(query_vectors, bad_document_vectors, dim=1)

    diff = good_similarity - bad_similarity
    return torch.max(torch.tensor(0), torch.tensor(margin) - diff).sum(dim=0)

def process_test_batch(tokenizer, batch, full_dataset, query_tower, doc_tower, device, margin):
    prepared = prepare_test_batch(tokenizer, batch, full_dataset, device)

    query_vectors = query_tower(prepared["queries"])              # tensor of shape (Q, E)
    good_document_vectors = doc_tower(prepared["good_documents"]) # tensor of shape (D, E)
    bad_document_vectors = doc_tower(prepared["bad_documents"])   # tensor of shape (D, E)

    # Duplicate each query so that it pairs with each document under that query
    # This means it ends up with the same length as the good/bad document vectors
    queries_for_each_document = torch.concat([ # tensor of shape (D, E) where D is the total number of documents
        query_vector.repeat(document_count, 1) # tensor of shape (K, E) where K is the number of documents for this query
        for (query_vector, document_count) in zip(query_vectors, prepared["document_count_for_each_query"])
    ])

    total_loss = calculate_triplet_loss(
        queries_for_each_document,
        good_document_vectors,
        bad_document_vectors,
        margin
    )

    return {
        "total_loss": total_loss,
        "document_count": good_document_vectors.shape[0],
    }


@dataclass
class DocumentTowerParameters:
    training: TrainingHyperparameters
    tokenizer: ModelTokenizer
    model: ModelHyperparameters

class DocumentTower(nn.Module):
    def __init__(self, parameters: DocumentTowerParameters):
        super(DocumentTower, self).__init__()
        self.model = PooledTowerModel(
            default_token_embeddings=parameters.tokenizer.default_token_embeddings,
            training_parameters=parameters.training,
            hidden_layer_sizes=parameters.model.doc_tower_hidden_dimensions,
            include_layer_norms=parameters.model.include_layer_norms,
            output_size=parameters.model.comparison_embedding_size,
        )

    def forward(self, queries):
        return self.model(queries)


@dataclass
class QueryTowerParameters:
    training: TrainingHyperparameters
    tokenizer: ModelTokenizer
    model: ModelHyperparameters

class QueryTower(nn.Module):
    def __init__(self, parameters: QueryTowerParameters):
        super(QueryTower, self).__init__()
        self.model = PooledTowerModel(
            default_token_embeddings=parameters.tokenizer.default_token_embeddings,
            training_parameters=parameters.training,
            hidden_layer_sizes=parameters.model.query_tower_hidden_dimensions,
            include_layer_norms=parameters.model.include_layer_norms,
            output_size=parameters.model.comparison_embedding_size,
        )

    def forward(self, queries):
        return self.model(queries)
    
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


    def forward(self, queries):
        flattened_tokens = queries["flattened_tokens"]
        offsets = queries["offsets"]

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

if __name__ == "__main__":
    print("Loading dataset...")
    dataset = load_dataset("microsoft/ms_marco", "v1.1")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    margin = 0.2  # Margin for triplet loss
    print(f'Using device: {device}')
    train_dataset = dataset["train"]
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=lambda x: x,  # Specify identity collate function (no magic batching which breaks)
    )

    tokenizer = ModelTokenizer.load()
    token_embeddings = tokenizer.default_token_embeddings
    print(f"Tokenizer loaded. Vocabulary size {token_embeddings.shape[0]}, Embedding size: {token_embeddings.shape[1]}")

    training_parameters = TrainingHyperparameters(
        batch_size=128,
        epochs=2,
        learning_rate=0.002,
        freeze_embeddings=True,
        dropout=0.3,
    )
    model_parameters = ModelHyperparameters(
        comparison_embedding_size=128,
        query_tower_hidden_dimensions=[128, 64],
        doc_tower_hidden_dimensions=[128, 64],
        include_layer_norms=True
    )

    doc_tower = DocumentTower(parameters=DocumentTowerParameters(
        training=training_parameters,
        model=model_parameters,
        tokenizer=tokenizer,
    )).to(device)
    query_tower = QueryTower(parameters=QueryTowerParameters(
        training=training_parameters,
        model=model_parameters,
        tokenizer=tokenizer,
    )).to(device)

    combined_parameters = list(doc_tower.parameters()) + list(query_tower.parameters())
    optimizer = optim.Adam(combined_parameters, lr=0.002)

    print_every = 100
    running_loss = 0.0
    running_samples = 0

    total_batches = len(train_data_loader)

    print("Beginning training...")
    for batch_idx, raw_batch in enumerate(train_data_loader):
        optimizer.zero_grad()
        batch_results = process_test_batch(tokenizer, raw_batch, train_dataset, query_tower, doc_tower, device, margin)
        loss = batch_results["total_loss"]
        running_samples += batch_results["document_count"]
        running_loss += loss.item()

        loss.backward()
        optimizer.step()

        batch_num = batch_idx + 1
        if batch_num % print_every == 0:
            print(f"Batch {batch_num}/{total_batches}, Average Loss: {(running_loss / running_samples):.4f}")
            running_loss = 0.0
            running_samples = 0

        
