import redis
import numpy as np
import torch
from model import PooledTwoTowerModel
from model.common import select_device
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.query import Query
import datasets
import sys

def to_float32_bytes(vec):
    arr = np.array(vec, dtype=np.float32)
    return arr.tobytes()

def create_index(r, dim, distance_metric="COSINE"):
    try:
        r.ft('doc_idx').info()
        print("Index already exists.")
    except:
        schema = [
            TextField("url"),
            TextField("passage"),
            VectorField("doc_embedding", "HNSW", {
                "TYPE": "FLOAT32",
                "DIM": dim,
                "DISTANCE_METRIC": distance_metric
            })
        ]
        r.ft('doc_idx').create_index(schema)
        print(f"Index created with DIM={dim}, DISTANCE_METRIC={distance_metric}")

def vectorize_and_store(r, model, documents):
    for i, doc in enumerate(documents):
        passage = doc["passage"]
        url = doc["url"]
        tokenized = model.tokenize_document(passage)
        embedding = model.embed_tokenized_documents([tokenized]).detach().cpu().numpy()[0]
        vec_bytes = to_float32_bytes(embedding)
        redis_doc = {
            "url": url,
            "passage": passage,
            "doc_embedding": vec_bytes
        }
        r.hset(f"doc:{i}", mapping=redis_doc)
        print(f"Stored doc:{i}")

def test_query_search(r, query_model, query_text, expected_dim):
    tokenized = query_model.tokenize_query(query_text)
    embedding = query_model.embed_tokenized_queries([tokenized]).detach().cpu().numpy()[0]
    if len(embedding) != expected_dim:
        print(f"Error: Query embedding dim {len(embedding)} != expected {expected_dim}")
        return
    query_vec = np.array(embedding, dtype=np.float32).tobytes()
    k = 3
    vector_field = "doc_embedding"
    base_query = f"*=>[KNN {k} @{vector_field} $vec as score]"
    q = Query(base_query).sort_by("score").return_fields("url", "passage", "score").paging(0, k).dialect(2)
    params = {"vec": query_vec}
    results = r.ft('doc_idx').search(q, params)
    print(f"Top results for query '{query_text}':")
    for doc in results.docs:
        print(f"  url: {doc.url}, score: {doc.score}")

if __name__ == "__main__":
    # Redis connection
    r = redis.Redis(host='localhost', port=6379, db=0)

    # Model loading
    device = select_device()
    model_name = 'fixed-boosted-word2vec-linear'  # adjust as needed
    model = PooledTwoTowerModel.load_for_evaluation(model_name, device)

    # Get embedding dimension
    test_vec = model.embed_tokenized_documents([model.tokenize_document("test")]).detach().cpu().numpy()[0]
    vector_dim = len(test_vec)

    # Create Redis index
    create_index(r, vector_dim, distance_metric="COSINE")

    # Load MS MARCO sample
    msmarco_dataset = datasets.load_dataset("microsoft/ms_marco", "v1.1", split="train")
    msmarco_documents = []
    num_docs = min(100000, len(msmarco_dataset))  # Ensure we do not exceed dataset size
    for i in range(num_docs):
        query_row = msmarco_dataset[i]
        passages = query_row["passages"]["passage_text"]
        if passages and len(passages) > 0:
            passage_text = passages[0]
            doc_id = f"msmarco_{i}"
            msmarco_documents.append({"url": doc_id, "passage": passage_text})
            if (i + 1) % 100 == 0:
                print(f"Loaded {i + 1} documents...")

    # Store document vectors
    vectorize_and_store(r, model, msmarco_documents)

    # Validation: Search with query tower
    if hasattr(model, "tokenize_query") and hasattr(model, "embed_tokenized_queries"):
        test_query = "What is the capital of France?"
        test_query_search(r, model, test_query, vector_dim)
    else:
        print("Query tower methods not found. Please implement 'tokenize_query' and 'embed_tokenized_queries' in your model.")
