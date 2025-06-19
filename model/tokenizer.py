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

class TokenizerBase:
    def tokenize(self, string):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def generate_default_embeddings(self, kind) -> torch.Tensor:
        """
        Generate default embeddings based on the kind specified. A kind of "zeros" must be supported, as a minimum.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def generate_default_embedding_boosts(self, kind) -> torch.Tensor:
        """
        Generate default token embedding boosts based on the kind specified. A kind of "ones" must be supported, as a minimum.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

@dataclass
class Word2VecTokenizer(TokenizerBase):
    words: list[str]
    token_map: dict[str, int]
    loaded_token_embeddings: torch.Tensor

    @classmethod
    def load(cls):
        folder = os.path.dirname(__file__)
        word_vectors = torch.load(folder + '/data/week-1-word2vec-word-vectors.pt')

        embeddings_shape = word_vectors["embeddings"].shape
        print(f"Word2Vec Tokenizer loaded. Vocabulary size {embeddings_shape[0]}, Embedding size: {embeddings_shape[1]}")

        return cls(
            words=word_vectors["vocabulary"],
            token_map={word: i for i, word in enumerate(word_vectors["vocabulary"])},
            loaded_token_embeddings=word_vectors["embeddings"],
        )
    
    def tokenize(self, string):
        filtered_title_words = re.sub(r'[^a-z0-9 ]', '', string.lower()).split()
        mapped_words = [self.token_map[word] for word in filtered_title_words if word in self.token_map]
        return mapped_words

    def generate_default_embeddings(self, kind):
        match kind:
            case "default":
                return self.loaded_token_embeddings
            case "random":
                return torch.randn_like(self.loaded_token_embeddings, dtype=torch.float32)
            case "zeros":
                return torch.zeros_like(self.loaded_token_embeddings, dtype=torch.float32)
            case _:
                raise ValueError(f"Unknown embedding kind: {kind}")
            
    def generate_default_embedding_boosts(self, kind):
        shape = [self.loaded_token_embeddings.shape[0]]
        match kind:
            case "ones":
                return torch.ones(shape, dtype=torch.float32)
            case "zeros":
                return torch.zeros(shape, dtype=torch.float32)
            case "sqrt-inverse-frequency":
                folder = os.path.dirname(__file__)
                word_counts = pd.read_csv(folder + '/data/week-1-word2vec-word-counts.csv')
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

                return torch.tensor([generate_shrink_factor(word) for word in self.words], dtype=torch.float32)
            case _:
                raise ValueError(f"Unknown embedding boost kind: {kind}")

@dataclass
class PretrainedTokenizer(TokenizerBase):
    tokenizer: transformers.PreTrainedTokenizer
    loaded_token_embeddings: torch.Tensor

    @classmethod
    def load(cls, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        print(f"Loading pretrained tokenizer and embeddings for {model_name}...")
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model: transformers.PreTrainedModel = transformers.AutoModel.from_pretrained(model_name)

        with torch.no_grad():
            embeddings = model.get_input_embeddings().weight

        return cls(
            tokenizer=tokenizer,
            loaded_token_embeddings=embeddings,
        )
    
    def tokenize(self, string):
        return self.tokenizer.encode(string)

    def generate_default_embeddings(self, kind):
        match kind:
            case "default":
                return self.loaded_token_embeddings
            case "random":
                return torch.randn_like(self.loaded_token_embeddings, dtype=torch.float32)
            case "zeros":
                return torch.zeros_like(self.loaded_token_embeddings, dtype=torch.float32)
            case _:
                raise ValueError(f"Unknown embedding kind: {kind}")

    def generate_default_embedding_boosts(self, kind):
        shape = [self.loaded_token_embeddings.shape[0]]
        match kind:
            case "ones":
                return torch.ones(shape, dtype=torch.float32)
            case "zeros":
                return torch.zeros(shape, dtype=torch.float32)
            case _:
                raise ValueError(f"Unknown embedding boost kind: {kind}")
            

def get_tokenizer(tokenizer_name: str) -> TokenizerBase:
    if tokenizer_name.startswith("pretrained:"):
        model_name = tokenizer_name[len("pretrained:"):]
        return PretrainedTokenizer.load(model_name=model_name)

    match tokenizer_name:
        case "week1-word2vec":
            return Word2VecTokenizer.load()
        case _:
            raise ValueError(f"Unknown tokenizer: {tokenizer_name}")