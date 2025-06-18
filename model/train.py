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

@dataclass
class TrainingHyperparameters:
    batch_size: int
    epochs: int
    learning_rate: float
    freeze_embeddings: bool
    dropout: float
    margin: float = 0.2  # Margin for triplet loss

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
        word_vectors = torch.load(folder + '/word_vectors.pt')

        # Implement boosting
        import pandas as pd
        import math
        word_counts = pd.read_csv(folder + '/word_counts.csv')
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

def prepare_test_batch(raw_batch, negative_samples):
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
        "tokenized_queries": queries,
        "tokenized_good_documents": good_documents,
        "tokenized_bad_documents": bad_documents,
        "document_count_for_each_query": document_count_for_each_query,
    }

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

def calculate_triplet_loss(query_vectors, good_document_vectors, bad_document_vectors, margin: float = 0.2):
    """
    Calculate the loss for a single query-document pair. 
    """
    good_similarity = F.cosine_similarity(query_vectors, good_document_vectors, dim=1)
    bad_similarity = F.cosine_similarity(query_vectors, bad_document_vectors, dim=1)

    diff = good_similarity - bad_similarity
    return torch.max(torch.tensor(0), torch.tensor(margin) - diff).sum(dim=0)

def process_test_batch(batch, negative_samples, query_tower, doc_tower, margin):
    prepared = prepare_test_batch(batch, negative_samples)

    query_vectors = query_tower(prepared["tokenized_queries"])              # tensor of shape (Q, E)
    good_document_vectors = doc_tower(prepared["tokenized_good_documents"]) # tensor of shape (D, E)
    bad_document_vectors = doc_tower(prepared["tokenized_bad_documents"])   # tensor of shape (D, E)

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
    tokenizer: Word2VecTokenizer
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
            batch_results = process_test_batch(raw_batch, self.negative_samples, query_tower, doc_tower, self.training_parameters.margin)
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
        print(f"Epoch {self.epoch} complete")

    def validate(self):
        print()
        print("== VALIDATING MODEL ==")
        print()
        document_passage_to_document_index_lookup = {} # passage => index
        document_index_to_document_ids = []
        document_index_to_passage_texts = []
        document_index_to_tokenized_doc = []
        total_documents = 0
    
        query_passage_to_query_index_lookup = {} # passage => query_index
        query_index_to_tokenized_query = []
        query_index_to_query_ids = []
        query_index_to_query_text = []

        print("Preparing validation data...")
        for query_row in self.validation_dataset:
            query_id = query_row['query_id']

            # Add passages
            for i, (tokenized_passage, passage_text) in enumerate(zip(query_row["tokenized_passages"], query_row["passages"]["passage_text"])):
                total_documents += 1
                if passage_text in document_passage_to_document_index_lookup:
                    document_index = document_passage_to_document_index_lookup[passage_text]
                else:
                    document_index = len(document_passage_to_document_index_lookup)
                    document_passage_to_document_index_lookup[passage_text] = document_index
                    document_index_to_document_ids.append([])
                    document_index_to_passage_texts.append(passage_text)
                    document_index_to_tokenized_doc.append(tokenized_passage)
    
                document_index_to_document_ids[document_index].append((query_id, i))

            query_text = query_row['query']
            if query_text in query_passage_to_query_index_lookup:
                query_index = query_passage_to_query_index_lookup[query_text]
            else:
                query_index = len(query_passage_to_query_index_lookup)
                query_passage_to_query_index_lookup[query_text] = i
                query_index_to_tokenized_query.append(query_row['tokenized_query'])
                query_index_to_query_ids.append([])
                query_index_to_query_text.append(query_text)

            query_index_to_query_ids[query_index].append(query_id)

        distinct_query_count = len(query_index_to_tokenized_query)
        distinct_document_count = len(document_index_to_tokenized_doc)
        print(f"Distinct queries: {distinct_query_count} (of {len(self.validation_dataset)} total queries)")
        print(f"Distinct documents: {distinct_document_count} (of {total_documents} total documents)")

        print("Generating embeddings for validation data...")
        self.doc_tower.eval()
        self.query_tower.eval()
        
        query_embeddings = self.query_tower(query_index_to_tokenized_query)
        document_embeddings = self.query_tower(document_index_to_tokenized_doc)

        average_relevance_by_query = []
        reciprical_ranks_of_first_relevant_result_by_query = []
        any_relevant_result_by_query = []

        k_samples = 5
        total_queries_to_consider = 1000
        query_to_print = random.randint(0, total_queries_to_consider - 1)

        print(f"Generating top-{k_samples} docs for {total_queries_to_consider} queries in validation set...")

        for sample_query_id, query_embedding in enumerate(query_embeddings[:total_queries_to_consider]):
            similarities = F.cosine_similarity(query_embedding, document_embeddings, dim=1).tolist() # (D) [5, 7, 2]
            top_k_most_similar = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[:k_samples]

            relevant_query_ids = set(query_index_to_query_ids[i])
            top_k_is_relevant = [
                any(document_id[0] in relevant_query_ids for document_id in document_index_to_document_ids[document_index])
                for document_index, _ in top_k_most_similar
            ]

            inverse_first_relevant_rank = 0
            for i in range(k_samples):
                if top_k_is_relevant[i]:
                    inverse_first_relevant_rank = 1 / (i + 1)
                    break
            reciprical_ranks_of_first_relevant_result_by_query.append(inverse_first_relevant_rank)
            average_relevance_by_query.append(statistics.mean(top_k_is_relevant))
            any_relevant_result_by_query.append(any(top_k_is_relevant))

            if sample_query_id == query_to_print:
                query_ids = query_index_to_query_ids[i]
                print()
                print(f"Example query {query_ids} \"{query_index_to_query_text[i]}\" top {k_samples} most similar documents:")
                for document_index, score in top_k_most_similar:
                    document_ids = document_index_to_document_ids[document_index]
                    passage = document_index_to_passage_texts[document_index]
                    print(f"  => {document_ids} with score {score:.3f}: \"{passage}\"")
                print()
        
        print(f"Across the first {total_queries_to_consider} queries, we queried the top {k_samples} documents (from {distinct_document_count} docs across {distinct_query_count} queries):")
        print(f"> Proportion with any relevant result     : {statistics.mean(any_relevant_result_by_query):.2%}")
        print(f"> Reciprical rank of first relevant result: {statistics.mean(reciprical_ranks_of_first_relevant_result_by_query):.2%}")
        print(f"> Average relevance of results            : {statistics.mean(average_relevance_by_query):.2%}")
        print()
        print(f"Validation complete")
        print()
    
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



        
