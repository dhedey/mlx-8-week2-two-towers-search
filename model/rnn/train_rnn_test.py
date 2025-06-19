from dataclasses import dataclass, field
import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import re
import os

@dataclass
class TrainingHyperparameters:
    batch_size: int
    epochs: int
    learning_rate: float
    freeze_embeddings: bool
    dropout: float
    margin: float = 0.2  # Margin for triplet loss
    validation_batch_size: int = 128  # Batch size for validation

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


class TokenizerBase:
    def embeddings(self):
        pass

    def tokenize(self, string):
        pass
@dataclass
class Word2VecTokenizer(TokenizerBase):
    token_map: dict[str, int]
    default_token_embeddings: torch.Tensor

    @classmethod
    def load(cls):
        folder = os.path.dirname(__file__)
        # Go up one directory to access the data folder
        data_folder = os.path.join(os.path.dirname(folder), 'data')
        word_vectors = torch.load(os.path.join(data_folder, 'week-1-word2vec-word-vectors.pt'), weights_only=False)

        # Implement boosting
        import pandas as pd
        import math
        word_counts = pd.read_csv(os.path.join(data_folder, 'week-1-word2vec-word-counts.csv'))
        word_counts = {
            row["word"]: row["count"]
            for index, row in word_counts.iterrows()
        }

        def generate_shrink_factor(word):
            MIN_APPEARANCE_THRESHOLD = 10

            if word in word_counts:
                appearance_count = max(MIN_APPEARANCE_THRESHOLD, word_counts[word])
            else:
                appearance_count = MIN_APPEARANCE_THRESHOLD
            
            inverse_freqency = 10 / appearance_count # Between 0 and 1

            return math.sqrt(inverse_freqency)
        
        print("Boosting embeddings...")
        # embeddings = word_vectors["embeddings"] * torch.diag(
        #     torch.tensor([generate_shrink_factor(word) for word in word_vectors["vocabulary"]])
        # )
        embeddings = torch.stack([
            word_embedding * generate_shrink_factor(word)
            for (word_embedding, word) in zip(word_vectors["embeddings"], word_vectors["vocabulary"])
        ])
        print("Boosting complete")

        return cls(
            token_map={word: i for i, word in enumerate(word_vectors["vocabulary"])},
            default_token_embeddings=embeddings,
        )

    def embeddings(self):
        return self.default_token_embeddings
    
    def tokenize(self, string):
        filtered_title_words = re.sub(r'[^a-z0-9 ]', '', string.lower()).split()
        mapped_words = [self.token_map[word] for word in filtered_title_words if word in self.token_map]
        return mapped_words

def prepare_test_batch(raw_batch, negative_samples, device):
    queries = []
    good_documents = []
    bad_documents = []
    document_count_for_each_query = []

    for query_row in raw_batch:
        queries.append(query_row["tokenized_query"])
        good_passages = query_row["tokenized_passages"]
        good_documents.extend(good_passages)
        
        good_passage_count = len(good_passages)
        document_count_for_each_query.append(good_passage_count)

    # TODO: Add in retrying of query-id clashes
    negative_sample_choices = torch.randint(low=0, high=len(negative_samples), size=(len(good_documents),)).tolist()
    bad_documents = [
        negative_samples[i]["tokenized_passage"] for i in negative_sample_choices
    ]
    
    return {
        "queries": prepare_tokens_for_rnn(queries, device),
        "good_documents": prepare_tokens_for_rnn(good_documents, device),
        "bad_documents": prepare_tokens_for_rnn(bad_documents, device),
        "document_count_for_each_query": document_count_for_each_query, # List not tensor
    }

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

def calculate_triplet_loss(query_vectors, good_document_vectors, bad_document_vectors, margin: float = 0.2):
    """
    Calculate the loss for a single query-document pair. 
    """
    good_similarity = F.cosine_similarity(query_vectors, good_document_vectors, dim=1)
    bad_similarity = F.cosine_similarity(query_vectors, bad_document_vectors, dim=1)

    diff = good_similarity - bad_similarity
    return torch.max(torch.tensor(0), torch.tensor(margin) - diff).sum(dim=0)

def process_test_batch(batch, negative_samples, query_tower, doc_tower, device, margin):
    prepared = prepare_test_batch(batch, negative_samples, device)

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
    tokenizer: Word2VecTokenizer
    model: ModelHyperparameters

class DocumentTower(nn.Module):
    def __init__(self, parameters: DocumentTowerParameters):
        super(DocumentTower, self).__init__()
        self.model = RNNTowerModel(
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
    tokenizer: Word2VecTokenizer
    model: ModelHyperparameters

class QueryTower(nn.Module):
    def __init__(self, parameters: QueryTowerParameters):
        super(QueryTower, self).__init__()
        self.model = RNNTowerModel(
            default_token_embeddings=parameters.tokenizer.default_token_embeddings,
            training_parameters=parameters.training,
            hidden_layer_sizes=parameters.model.query_tower_hidden_dimensions,
            include_layer_norms=parameters.model.include_layer_norms,
            output_size=parameters.model.comparison_embedding_size,
        )

    def forward(self, queries):
        return self.model(queries)
    
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
    
class ModelTrainer:
    def __init__(self, query_tower: torch.nn.Module, doc_tower: torch.nn.Module, training_parameters: TrainingHyperparameters, device):
        self.query_tower = query_tower
        self.doc_tower = doc_tower
        self.training_parameters = training_parameters
        self.device = device

        print("Preparing model for training...")
        combined_parameters = list(self.doc_tower.parameters()) + list(self.query_tower.parameters())
        self.optimizer = optim.Adam(combined_parameters, lr=0.002)

        print("Tokenizing queries and passages...")
        train_dataset = dataset["train"].map(
            lambda x: {
                "tokenized_query": tokenizer.tokenize(x["query"]),
                "tokenized_passages": [
                    tokenizer.tokenize(passage) for passage in x["passages"]["passage_text"]
                ],
            })

        self.train_data_loader = DataLoader(
            train_dataset,
            batch_size=training_parameters.batch_size,
            shuffle=True,
            collate_fn=lambda x: x,  # Specify identity collate function (no magic batching which breaks)
        )

        self.validation_dataset = dataset["validation"].map(
            lambda x: {
                "tokenized_query": tokenizer.tokenize(x["query"]),
                "tokenized_passages": [
                    tokenizer.tokenize(passage) for passage in x["passages"]["passage_text"]
                ],
            })
        self.validation_data_loader = DataLoader(
            self.validation_dataset,
            batch_size=training_parameters.batch_size,
            collate_fn=lambda x: x,  # Specify identity collate function (no magic batching which breaks)
        )

        print(f"Generating negative samples...")
        self.negative_samples = [
            { "tokenized_passage": tokenized_passage, "query_id": query_row["query_id"] }
            for query_row in train_dataset
            for tokenized_passage in query_row["tokenized_passages"]
        ]

    def train(self):
        print("Beginning training...")

        self.epoch = 1

        while self.epoch <= self.training_parameters.epochs:
            print(f"Epoch {self.epoch}/{self.training_parameters.epochs}")
            self.train_epoch()
            self.validate()
            self.save_model()
            self.epoch += 1

        print("Training complete.")

    def train_epoch(self):
        self.doc_tower.train()
        self.query_tower.train()

        print_every = 10
        running_loss = 0.0
        running_samples = 0
        total_batches = len(self.train_data_loader)
        for batch_idx, raw_batch in enumerate(self.train_data_loader):
                
            self.optimizer.zero_grad()
            batch_results = process_test_batch(raw_batch, self.negative_samples, query_tower, doc_tower, device, self.training_parameters.margin)
            loss = batch_results["total_loss"]
            running_samples += batch_results["document_count"]
            running_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            batch_num = batch_idx + 1
            if batch_num % print_every == 0:
                print(f"Epoch {self.epoch}, Batch {batch_num}/{total_batches}, Average Loss: {(running_loss / running_samples):.4f}")
                running_loss = 0.0
                running_samples = 0
        print(f"Epoch {self.epoch} complete (processed {total_batches} batches)")

    def validate_rnn(self, validation_batch_size=None):
        print("Validating model with batched processing:")
        
        # Use provided batch size or default from training parameters
        if validation_batch_size is None:
            validation_batch_size = self.training_parameters.validation_batch_size
        
        self.doc_tower.eval()
        self.query_tower.eval()
        
        # Create a custom DataLoader with the specified batch size
        validation_data_loader = DataLoader(
            self.validation_dataset,
            batch_size=validation_batch_size,
            collate_fn=lambda x: x,
        )
        
        total_queries_processed = 0
        total_recall_accuracy = 0.0
        k_samples = 5
        
        print(f"Processing validation queries and documents together in batches of {validation_batch_size}...")
        
        # Process each batch of queries and their documents together
        for batch_idx, raw_batch in enumerate(validation_data_loader):
                
            # Prepare batch data
            batch_query_tokens = []
            batch_query_ids = []
            batch_query_texts = []
            batch_document_tokens = []
            batch_document_ids = []
            batch_document_texts = []
            
            for query_row in raw_batch:
                    
                # Add query
                batch_query_tokens.append(query_row["tokenized_query"])
                batch_query_ids.append(query_row["query_id"])
                batch_query_texts.append(query_row["query"])
                
                # Add all documents for this query
                for i, passage in enumerate(query_row["tokenized_passages"]):
                    batch_document_tokens.append(passage)
                    batch_document_ids.append((query_row["query_id"], i))
                    batch_document_texts.append(query_row["passages"]["passage_text"][i])
            
            # Process queries and documents for this batch
            if batch_query_tokens and batch_document_tokens:
                query_embeddings = self.query_tower(
                    prepare_tokens_for_rnn(batch_query_tokens, self.device)
                )
                document_embeddings = self.doc_tower(
                    prepare_tokens_for_rnn(batch_document_tokens, self.device)
                )
            else:
                continue
            
            # Calculate similarities for each query against documents in this batch
            for query_idx, query_embedding in enumerate(query_embeddings):
                # Calculate similarities with documents in this batch
                similarities = F.cosine_similarity(
                    query_embedding.unsqueeze(0), 
                    document_embeddings,  # Documents from this batch only
                    dim=1
                ).tolist()
                
                # Get top k most similar documents
                top_k_most_similar = sorted(
                    enumerate(similarities), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:k_samples]
                
                query_id = batch_query_ids[query_idx]
                query_text = batch_query_texts[query_idx]
                
                # Count matching documents
                matching_document_count = 0
                for doc_idx, score in top_k_most_similar:
                    doc_query_id = batch_document_ids[doc_idx][0]
                    if doc_query_id == query_id:
                        matching_document_count += 1
                
                # Calculate recall for this query
                recall = matching_document_count / k_samples
                total_recall_accuracy += recall
                total_queries_processed += 1
                
                # Print sample results every 500 queries
                if total_queries_processed % 500 == 0:
                    print(f"Query {query_id} \"{query_text}\" top 3 most similar documents:")
                    for doc_idx, score in top_k_most_similar[:3]:
                        doc_id = batch_document_ids[doc_idx]
                        doc_text = batch_document_texts[doc_idx]
                        print(f"  => {doc_id} with score {score:.3f}: \"{doc_text}\"")
                    print(f"  Recall: {recall:.2%}")
                    print()
            
            # Clear memory for this batch
            del query_embeddings, document_embeddings, similarities
        
        # Calculate final metrics
        if total_queries_processed > 0:
            average_recall = total_recall_accuracy / total_queries_processed
            print(f"Validation complete")
            print(f"Processed {total_queries_processed} queries")
            print(f"Average top-{k_samples} recall: {average_recall:.2%}")
        else:
            print("No queries processed during validation")
        print()

    def validate(self):
        # Keep the old validate method for compatibility, but call the new one
        self.validate_rnn()

    def save_model(self):
        model_folder = os.path.dirname(__file__)
        model_filename = 'model.pt'
        torch.save({
            "query_tower": self.query_tower.state_dict(),
            "doc_tower": self.doc_tower.state_dict(),
            "training_parameters": self.training_parameters.to_dict(),
            "model_parameters": ModelHyperparameters().to_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }, os.path.join(model_folder, model_filename))
        print(f"Model saved to {model_filename}.")

if __name__ == "__main__":
    print("Loading dataset...")
    datasets.config.IN_MEMORY_MAX_SIZE = 8 * 1024 * 1024 # 8GB
    dataset = datasets.load_dataset("microsoft/ms_marco", "v1.1")
    DEVICE_IF_MPS_SUPPORT = 'cpu' # or 'mps' - but it doesn't work well with EmbeddingBag
    device = torch.device('cuda' if torch.cuda.is_available() else DEVICE_IF_MPS_SUPPORT if torch.backends.mps.is_available() else 'cpu')
    
    print(f'Using device: {device}')

    tokenizer = Word2VecTokenizer.load()
    token_embeddings = tokenizer.default_token_embeddings
    print(f"Tokenizer loaded. Vocabulary size {token_embeddings.shape[0]}, Embedding size: {token_embeddings.shape[1]}")

    training_parameters = TrainingHyperparameters(
        batch_size=128,
        epochs=20,
        learning_rate=0.002,
        freeze_embeddings=False,
        dropout=0.3,
    )
    model_parameters = ModelHyperparameters(
        comparison_embedding_size=64,
        # query_tower_hidden_dimensions=[128, 64],
        query_tower_hidden_dimensions=[],
        # doc_tower_hidden_dimensions=[128, 64],
        doc_tower_hidden_dimensions=[],
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

    trainer = ModelTrainer(
        query_tower=query_tower,
        doc_tower=doc_tower,
        training_parameters=training_parameters,
        device=device,
    )
    trainer.train()



        
