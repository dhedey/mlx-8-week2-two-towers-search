import streamlit as st
import redis
import os
import numpy as np
from typing import List, Dict, Tuple

# Redis connection
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', 6379))

st.title("Document Search Interface")

# Initialize Redis connection
try:
    redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    st.success(f"Successfully connected to Redis at {redis_host}:{redis_port}")
except redis.ConnectionError as e:
    st.error(f"Failed to connect to Redis: {str(e)}")
    redis_client = None

if redis_client:
    # Document Search Section
    st.header("Document Search")

    # Dummy Data Management
    st.subheader("Dummy Data Management")

    # Function to generate dummy embeddings
    def generate_dummy_embedding() -> List[float]:
        return np.random.randn(128).tolist()  # 128-dimensional embedding

    # Function to store dummy document
    def store_dummy_document(doc_id: str, content: str, embedding: List[float]):
        # Store document content
        redis_client.hset(f"doc:{doc_id}", mapping={
            "content": content,
            "embedding": str(embedding)  # Convert list to string for storage
        })

    # Function to get all documents
    def get_all_documents() -> List[Dict]:
        docs = []
        for key in redis_client.keys("doc:*"):
            doc_data = redis_client.hgetall(key)
            if doc_data:
                doc_id = key.split(":")[1]
                embedding = eval(doc_data["embedding"])  # Convert string back to list
                docs.append({
                    "id": doc_id,
                    "content": doc_data["content"],
                    "embedding": embedding
                })
        return docs

    # Function to search documents
    def search_documents(query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        docs = get_all_documents()
        if not docs:
            return []

        # Calculate cosine similarity
        similarities = []
        for doc in docs:
            similarity = np.dot(query_embedding, doc["embedding"]) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc["embedding"])
            )
            similarities.append((doc["id"], doc["content"], similarity))

        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:top_k]

    # UI for adding dummy documents
    st.write("Add Dummy Documents")
    col1, col2 = st.columns(2)

    with col1:
        doc_id = st.text_input("Document ID:", "doc1")
        doc_content = st.text_area("Document Content:", "This is a sample document about machine learning.")

    with col2:
        if st.button("Add Dummy Document"):
            try:
                embedding = generate_dummy_embedding()
                store_dummy_document(doc_id, doc_content, embedding)
                st.success(f"Successfully added document {doc_id}")
            except Exception as e:
                st.error(f"Error adding document: {str(e)}")

    # Search Interface
    st.subheader("Search Documents")
    search_query = st.text_input("Enter your search query:", "machine learning")

    if st.button("Search"):
        try:
            # Generate a dummy embedding for the query (in a real app, this would come from your model)
            query_embedding = generate_dummy_embedding()

            # Search documents
            results = search_documents(query_embedding)

            if results:
                st.write("Search Results:")
                for doc_id, content, similarity in results:
                    st.write(f"---")
                    st.write(f"Document ID: {doc_id}")
                    st.write(f"Content: {content}")
                    st.write(f"Similarity Score: {similarity:.4f}")
            else:
                st.info("No documents found. Add some documents first!")

        except Exception as e:
            st.error(f"Error during search: {str(e)}")

    # Display all stored documents
    st.subheader("Stored Documents")
    if st.button("Show All Documents"):
        docs = get_all_documents()
        if docs:
            for doc in docs:
                st.write(f"---")
                st.write(f"Document ID: {doc['id']}")
                st.write(f"Content: {doc['content']}")
        else:
            st.info("No documents stored yet.")

    # Cleanup Section
    st.header("Cleanup")
    if st.button("Clear All Documents"):
        try:
            for key in redis_client.keys("doc:*"):
                redis_client.delete(key)
            st.success("Successfully cleared all documents")
        except Exception as e:
            st.error(f"Error during cleanup: {str(e)}")