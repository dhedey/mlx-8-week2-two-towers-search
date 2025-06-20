import redis
import csv
import numpy as np
import torch
from model import PooledTwoTowerModel
from model.common import select_device
from redis.commands.search.field import TextField, VectorField

# Redis connection
r = redis.Redis(host='localhost', port=6379, db=0)

# Model loading
device = select_device()
model_name = 'fixed-boosted-word2vec-linear'  # adjust as needed
model = PooledTwoTowerModel.load_for_evaluation(model_name, device)

def to_float32_bytes(vec):
    arr = np.array(vec, dtype=np.float32)
    return arr.tobytes()

# Create RediSearch index if not exists
def create_index(vector_dim):
    try:
        r.ft('doc_idx').info()
        print("Index already exists.")
    except:
        schema = [
            TextField("url"),
            TextField("passage"),
            VectorField("vector-two-tower-boosted-word2vec-linear", "HNSW", {
                "TYPE": "FLOAT32",
                "DIM": vector_dim,
                "DISTANCE_METRIC": "COSINE"
            })
        ]
        r.ft('doc_idx').create_index(schema)
        print("Index created.")

# Load and vectorize documents
def vectorize_and_store(csv_path, limit=100):
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            passage = row.get('passage', '') or row.get('text', '') or str(row)
            url = row.get('url', f'doc_{i}')
            tokenized = model.tokenize_query(passage)
            embedding = model.embed_tokenized_queries([tokenized]).detach().cpu().numpy()[0]
            vec_bytes = to_float32_bytes(embedding)
            doc = {
                "url": url,
                "passage": passage,
                "vector-two-tower-boosted-word2vec-linear": vec_bytes
            }
            r.hset(f"doc:{i}", mapping=doc)
            print(f"Stored doc:{i}")

if __name__ == "__main__":
    # Get embedding dimension
    test_vec = model.embed_tokenized_queries([model.tokenize_query("test")]).detach().cpu().numpy()[0]
    vector_dim = len(test_vec)
    create_index(vector_dim)
    vectorize_and_store("model/data/week-1-word2vec-word-counts.csv", limit=100)