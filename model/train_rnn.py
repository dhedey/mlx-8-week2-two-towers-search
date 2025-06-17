import torch
import torch.nn as nn
import torch.nn.functional as F

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

def neural_network(token_embedding, hidden_state) -> torch.Tensor:
    # Add batch dimension if not present
    if token_embedding.dim() == 1:
        token_embedding = token_embedding.unsqueeze(0)  # Add batch dimension
    if hidden_state.dim() == 1:
        hidden_state = hidden_state.unsqueeze(0)  # Add batch dimension
    
    # Now concatenate along dimension 1
    combined = torch.cat([token_embedding, hidden_state], dim=1)
    
    # Create a hidden layer
    # input_size is the sum of token_embedding and hidden_state dimensions
    # output_size is the same as hidden_state dimension
    hidden_layer = HiddenLayer(
        input_size=combined.shape[1],  # total size of concatenated input
        output_size=hidden_state.shape[1],  # same as hidden state size
        include_layer_norm=True,  # use layer normalization
        dropout=0.2  # 20% dropout
    )
    
    # Process through the hidden layer
    new_hidden_state = hidden_layer(combined)
    return new_hidden_state

def RNN(token_embeddings, initial_hidden_state) -> torch.Tensor:
    hidden_state = initial_hidden_state
    for token_embedding in token_embeddings:
        hidden_state = neural_network(token_embedding, hidden_state)
    final_hidden_state = hidden_state
    return final_hidden_state

if __name__ == "__main__":

    # Example usage with batch size 32
    batch_size = 32
    sequence_length = 10
    embedding_size = 10
    hidden_size = 10
    # Create tensors with proper dimensions
    token_embeddings = torch.randn(sequence_length, embedding_size)  # (sequence_length, embedding_size)
    initial_hidden_state = torch.randn(batch_size, hidden_size)  # (batch_size, hidden_size)
    
    print("Input shapes:")
    print(f"token_embeddings: {token_embeddings.shape}")
    print(f"initial_hidden_state: {initial_hidden_state.shape}")
    
    final_hidden_state = RNN(token_embeddings, initial_hidden_state)
    print(f"final_hidden_state: {final_hidden_state.shape}")