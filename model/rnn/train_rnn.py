import torch
import torch.nn as nn
import re

queries = [{
    "query_id": "123",
    "query_text": "What is the capital of France?",
}, {
    "query_id": "124",
    "query_text": "What is the capital of Germany? ",
}]
good_passages = [{
    "passage_id": "456",
    "passage_text": "France is a country in Europe.",
}, {
    "passage_id": "457",
    "passage_text": "Paris is the capital of France.",
}, {
    "passage_id": "458",
    "passage_text": "Paris is the biggest city in France.",
}]
bad_passages = [{
    "passage_id": "789",
    "passage_text": "I like to eat pizza.",
}, {
    "passage_id": "101",
    "passage_text": "London is a very big city.",
}, {
    "passage_id": "102",
    "passage_text": "I like to eat pasta.",
}]

triplet_structure = {
    "query_id": "123",
    "good_passage_id": "456",
    "bad_passage_id": "789",
}

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


class DocumentTower(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size):
        super(DocumentTower, self).__init__()
        self.model = RNNTowerModel(vocab_size, embedding_size, hidden_size, output_size)

    def forward(self, tokenized_passages):
        return self.model(tokenized_passages)

class QueryTower(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size):
        super(QueryTower, self).__init__()
        self.model = RNNTowerModel(vocab_size, embedding_size, hidden_size, output_size)

    def forward(self, tokenized_queries):
        return self.model(tokenized_queries)

class RNNTowerModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size):
        super(RNNTowerModel, self).__init__()
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=embedding_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, tokenized_queries):
        # tokenized_queries is a list of lists of token IDs
        # e.g., [[1, 2, 3], [4, 5, 6, 7]]
        
        # Pad sequences to the same length
        max_length = max(len(seq) for seq in tokenized_queries)
        padded_queries = []
        
        for seq in tokenized_queries:
            # Pad with 0 (assuming 0 is padding token)
            padded_seq = [0] * (max_length - len(seq)) + seq  # [0,0,1,2,4]
            print(padded_seq)
            padded_queries.append(padded_seq)
        
        # Convert to tensor
        query_tensor = torch.tensor(padded_queries, dtype=torch.long)
        
        # Get embeddings for each token
        embedded_queries = self.embeddings(query_tensor)  # Shape: [batch_size, seq_len, embedding_size]
        
        # Process through RNN
        rnn_output, hidden = self.rnn(embedded_queries)  # Shape: [batch_size, seq_len, hidden_size]
        
        # Take the last hidden state for each sequence
        last_hidden = rnn_output[:, -1, :]  # Shape: [batch_size, hidden_size]
        
        # Pass through additional layers
        x = self.hidden_layer(last_hidden)
        x = self.output_layer(x)
        
        return x

# Simple tokenizer function
def tokenize_query(query_text):
    # Simple word-based tokenization
    words = re.sub(r'[^a-z0-9 ]', '', query_text.lower()).split()
    # Create a simple vocabulary mapping (in practice, you'd use a proper tokenizer)
    vocab = {"what": 1, "is": 2, "the": 3, "capital": 4, "of": 5, "france": 6, "germany": 7}
    return [vocab.get(word, 0) for word in words if word in vocab]

# Tokenize the queries
tokenized_queries = [tokenize_query(q["query_text"]) for q in queries]
print("Tokenized queries:", tokenized_queries)

# Initialize model
vocab_size = 10  # Size of vocabulary
embedding_size = 100
hidden_size = 100
output_size = 100
doc_tower = DocumentTower(vocab_size, embedding_size, hidden_size, output_size)
query_tower = QueryTower(vocab_size, embedding_size, hidden_size, output_size)

# Process queries
query_embeddings = query_tower(tokenized_queries)
print("Query embeddings shape:", query_embeddings.shape)
print("Query embeddings:", query_embeddings)

#train the model
epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for query in tokenized_queries:
        query_embedding = query_tower(query)
        print("Query embedding:", query_embedding)
        break


#validate the model

#save the model