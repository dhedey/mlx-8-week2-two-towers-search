import streamlit as st
import redis
import os
import numpy as np
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import models
from model.common import TrainingHyperparameters

# Constants
DEFAULT_EMBEDDING_SIZE = 128
DEFAULT_TOP_K = 5

# Initialize model parameters
MODEL_PARAMETERS = models.PooledTwoTowerModelHyperparameters(
    tokenizer="week1-word2vec",
    comparison_embedding_size=DEFAULT_EMBEDDING_SIZE,
    query_tower_hidden_dimensions=[256, 128],
    doc_tower_hidden_dimensions=[256, 128],
    include_layer_norms=True
)

# Initialize training parameters
TRAINING_PARAMETERS = TrainingHyperparameters(
    initial_token_embeddings_kind="random",
    freeze_embeddings=True,
    dropout=0.1,
    batch_size=32,
    epochs=1,
    learning_rate=0.001,
    margin=0.1
)

class SearchApp:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            decode_responses=True
        )
        self.model = models.PooledTwoTowerModel(
            training_parameters=TRAINING_PARAMETERS,
            model_parameters=MODEL_PARAMETERS
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def search(self, query: str, top_k: int = DEFAULT_TOP_K):
        # Tokenize and embed the query
        with torch.no_grad():
            tokens = self.model.tokenize_query(query)
            tokenized_queries = [tokens]
            query_embedding = self.model.embed_tokenized_queries(tokenized_queries)
            try:
                # Add debug prints
                print(f"Query embedding type: {type(query_embedding)}")
                print(f"Query embedding shape: {query_embedding.shape}")
                print(f"Query embedding device: {query_embedding.device}")

                # Ensure tensor is on CPU before numpy conversion
                query_embedding = query_embedding[0].cpu()
                print(f"After CPU conversion - type: {type(query_embedding)}")

                # Convert to numpy
                query_embedding = query_embedding.numpy()
                print(f"After numpy conversion - type: {type(query_embedding)}")
            except Exception as e:
                print(f"Error during tensor conversion: {str(e)}")
                raise

        # Get all documents from Redis
        docs = []
        for key in self.redis_client.keys("doc:*"):
            doc_data = self.redis_client.hgetall(key)
            if doc_data:
                doc_embedding = np.array(eval(doc_data["embedding"]))
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                docs.append((key, doc_data["content"], similarity))

        # Sort by similarity and return top k
        docs.sort(key=lambda x: x[2], reverse=True)
        return docs[:top_k]

def main():
    st.title("Document Search")

    # Initialize the search app
    app = SearchApp()

    # Model selection dropdown (currently only one option)
    st.selectbox(
        "Select Model",
        ["PooledTwoTowerModel"],
        disabled=True
    )

    # Search query input
    query = st.text_input("Enter your search query:")

    # Search button
    if st.button("Search") and query:
        results = app.search(query, DEFAULT_TOP_K)

        # Display results
        if results:
            st.subheader("Search Results")
            for i, (doc_id, content, similarity) in enumerate(results, 1):
                st.markdown(f"**Result {i}** (Similarity: {similarity:.3f})")
                st.markdown(f"Document ID: {doc_id}")
                st.markdown(f"Content: {content}")
                st.markdown("---")
        else:
            st.info("No results found.")

if __name__ == "__main__":
    main()