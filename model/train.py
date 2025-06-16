from dataclasses import dataclass
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import random
    
VOCAB_SIZE = 60000  # Example vocabulary size, adjust as needed

def tokenize(string):
    token_count = random.randint(1, 20)
    return [random.randint(0, VOCAB_SIZE - 1) for _ in range(token_count)]

def prepare_test_batch(raw_batch, full_dataset, device):
    queries = []
    good_documents = []
    bad_documents = []
    dataset_len = len(full_dataset)
    document_count_for_each_query = []

    if dataset_len <= 1:
        raise ValueError("Dataset must contain more than one item to sample bad documents.")

    for query_row in raw_batch:
        queries.append(query_row["query"])
        good_passages = query_row["passages.passage_text"]
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
             
            random_passage = random.sample(random_row["passages.passage_text"])[0]
            bad_documents.append(random_passage)
    
    return {
        "queries": prepare_tokens_for_embedding_bag(queries, tokenize, device),
        "good_documents": prepare_tokens_for_embedding_bag(good_documents, tokenize, device),
        "bad_documents": prepare_tokens_for_embedding_bag(bad_documents, tokenize, device),
        "document_count_for_each_query": document_count_for_each_query, # List not tensor
    }

def prepare_tokens_for_embedding_bag(strings, tokenize_method, device):
    """
    See https://docs.pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html
    """
    flattened_tokens = []
    offsets = []
    current_offset = 0

    for string in strings:
        offsets.append(current_offset)
        tokens = tokenize_method(string)
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
    return torch.max(torch.tensor(0.0), torch.tensor(margin) - diff, dim=1).sum(dim=0)

def process_test_batch(batch, full_dataset, query_tower, doc_tower, device, margin):
    prepared = prepare_test_batch(batch, full_dataset, device)

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

class DocumentTower(nn.Module):
    def __init__(self):
        super(DocumentTower, self).__init__()
        self.average_pooling = nn.EmbeddingBag(VOCAB_SIZE, 128, mode='mean')

    def forward(self, documents):
        flattened_tokens = documents["flattened_tokens"]
        offsets = documents["offsets"]

        return self.average_pooling(flattened_tokens, offsets)
    

class QueryTower(nn.Module):
    def __init__(self):
        super(QueryTower, self).__init__()
        self.average_pooling = nn.EmbeddingBag(VOCAB_SIZE, 128, mode='mean')

    def forward(self, queries):
        flattened_tokens = queries["flattened_tokens"]
        offsets = queries["offsets"]

        return self.average_pooling(flattened_tokens, offsets)

if __name__ == "__main__":
    dataset = load_dataset("microsoft/ms_marco", "v1.1")
    train_data_loader = DataLoader(dataset["train"], batch_size=2, shuffle=True)

    doc_tower = DocumentTower()
    query_tower = QueryTower()
    combined_parameters = list(doc_tower.parameters()) + list(query_tower.parameters())

    optimizer = optim.Adam(combined_parameters, lr=0.002)

    print_every = 100
    running_loss = 0.0
    running_samples = 0

    total_batches = len(train_data_loader)
    for batch_idx, raw_batch in enumerate(train_data_loader):
        optimizer.zero_grad()
        batch_results = process_test_batch(raw_batch)
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

        
