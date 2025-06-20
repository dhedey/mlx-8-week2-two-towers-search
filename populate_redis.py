"""
Script to populate Redis with document passages and their vector representations
using the same dataset and model pipeline as in training.

- Loads the MS MARCO dataset using the datasets library
- Loads the trained two-tower model
- Vectorizes each document passage with the two-tower model
- Stores each record in Redis as a hash: {url, passage, vector-<model-name>}

Requirements: redis, torch, datasets
"""
import redis
import torch
import json
from model.models import load_model_for_evaluation
from model.trainer import TrainingHyperparameters
import datasets

REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0
TWO_TOWER_MODEL = "two-tower-boosted-word2vec-linear"

# Connect to Redis
db = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

def main():
    # Delete all existing keys in the Redis DB
    db.flushdb()
    print("Flushed all existing keys from Redis DB.")

    # Load two-tower model for evaluation
    two_tower_model = load_model_for_evaluation(TWO_TOWER_MODEL)
    two_tower_model.eval()

    # Load MS MARCO dataset
    dataset = datasets.load_dataset("microsoft/ms_marco", "v1.1")
    train_dataset = dataset["train"]

    two_tower_field = f'vector-{TWO_TOWER_MODEL}'

    # Process all records in the dataset
    for idx, row in enumerate(train_dataset):
        # Each row has: 'query', 'query_id', 'passages' (dict with 'passage_text', 'url', ...)
        for passage, url in zip(row['passages']['passage_text'], row['passages']['url']):
            # Vectorize passage with two-tower model
            tokenized = two_tower_model.tokenize_document(passage)
            two_tower_vector = two_tower_model.embed_tokenized_documents([tokenized])[0].detach().cpu().numpy().tolist()
            # Store in Redis (use url+idx as key to avoid collisions)
            key = f"doc:{url}:{idx}"
            db.hset(key, mapping={
                'url': url,
                'passage': passage,
                two_tower_field: json.dumps(two_tower_vector),
            })
    print(f"Populated Redis with all document passages using two-tower model.")

if __name__ == "__main__":
    main()
