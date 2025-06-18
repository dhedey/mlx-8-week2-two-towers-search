import streamlit as st
import redis
import os
import numpy as np
from typing import List, Dict, Tuple, Optional
import ast
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_EMBEDDING_SIZE = 128
DEFAULT_TOP_K = 5

# Model configurations
MODEL_OPTIONS = {
    "random": "Random Embeddings (Placeholder)",
    "future_model_1": "Future Model 1 (Coming Soon)",
    "future_model_2": "Future Model 2 (Coming Soon)"
}


class RedisManager:
    """Handles Redis connection and operations"""

    def __init__(self, host: str = None, port: int = None):
        self.host = host or os.getenv('REDIS_HOST', 'localhost')
        self.port = port or int(os.getenv('REDIS_PORT', 6379))
        self.client = None
        self._connect()

    def _connect(self) -> bool:
        """Establish Redis connection"""
        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            self.client.ping()
            return True
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.client = None
            return False

    def is_connected(self) -> bool:
        """Check if Redis connection is active"""
        return self.client is not None

    def get_connection_status(self) -> str:
        """Get connection status message"""
        if self.is_connected():
            return f"Successfully connected to Redis at {self.host}:{self.port}"
        return f"Failed to connect to Redis at {self.host}:{self.port}"


class EmbeddingService:
    """Handles embedding generation"""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for given text"""
        if self.model_name == "random":
            return np.random.randn(DEFAULT_EMBEDDING_SIZE).tolist()
        else:
            # Placeholder for future models
            return np.random.randn(DEFAULT_EMBEDDING_SIZE).tolist()

    def get_model_info(self) -> str:
        """Get model information"""
        return MODEL_OPTIONS.get(self.model_name, "Unknown Model")


class DocumentStore:
    """Handles document storage and retrieval"""

    def __init__(self, redis_manager: RedisManager):
        self.redis_manager = redis_manager

    def store_document(self, doc_id: str, content: str, embedding: List[float]) -> bool:
        """Store document with embedding"""
        try:
            if not self.redis_manager.is_connected():
                return False

            self.redis_manager.client.hset(f"doc:{doc_id}", mapping={
                "content": content,
                "embedding": str(embedding)
            })
            return True
        except Exception as e:
            logger.error(f"Error storing document {doc_id}: {e}")
            return False

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Retrieve a single document"""
        try:
            if not self.redis_manager.is_connected():
                return None

            doc_data = self.redis_manager.client.hgetall(f"doc:{doc_id}")
            if not doc_data:
                return None

            return {
                "id": doc_id,
                "content": doc_data["content"],
                "embedding": ast.literal_eval(doc_data["embedding"])
            }
        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {e}")
            return None

    def get_all_documents(self) -> List[Dict]:
        """Retrieve all documents"""
        try:
            if not self.redis_manager.is_connected():
                return []

            docs = []
            for key in self.redis_manager.client.keys("doc:*"):
                doc_id = key.split(":")[1]
                doc = self.get_document(doc_id)
                if doc:
                    docs.append(doc)
            return docs
        except Exception as e:
            logger.error(f"Error retrieving all documents: {e}")
            return []

    def delete_all_documents(self) -> bool:
        """Delete all documents"""
        try:
            if not self.redis_manager.is_connected():
                return False

            keys = self.redis_manager.client.keys("doc:*")
            if keys:
                self.redis_manager.client.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False


class SearchEngine:
    """Handles document search operations"""

    def __init__(self, document_store: DocumentStore):
        self.document_store = document_store

    def search_documents(self, query_embedding: List[float], top_k: int = DEFAULT_TOP_K) -> List[Tuple[str, str, float]]:
        """Search documents using cosine similarity"""
        docs = self.document_store.get_all_documents()
        if not docs:
            return []

        similarities = []
        query_norm = np.linalg.norm(query_embedding)

        for doc in docs:
            try:
                doc_embedding = doc["embedding"]
                doc_norm = np.linalg.norm(doc_embedding)

                if query_norm == 0 or doc_norm == 0:
                    similarity = 0.0
                else:
                    similarity = np.dot(query_embedding, doc_embedding) / (query_norm * doc_norm)

                similarities.append((doc["id"], doc["content"], similarity))
            except Exception as e:
                logger.error(f"Error calculating similarity for doc {doc['id']}: {e}")
                continue

        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:top_k]


class DocumentSearchApp:
    """Main application class"""

    def __init__(self):
        self.redis_manager = RedisManager()
        self.document_store = DocumentStore(self.redis_manager)
        self.search_engine = SearchEngine(self.document_store)
        self.embedding_service = None

    def setup_ui(self):
        """Setup the Streamlit UI"""
        st.title("Document Search Interface")

        # Connection status
        if self.redis_manager.is_connected():
            # st.success(self.redis_manager.get_connection_status())
            pass
        else:
            st.error(self.redis_manager.get_connection_status())
            st.stop()

        # Model selection
        self._render_model_selection()

        # Document management
        self._render_document_management()

        # Search interface
        self._render_search_interface()

        # Document display
        self._render_document_display()

        # Cleanup section
        self._render_cleanup_section()

    def _render_model_selection(self):
        """Render model selection section"""
        st.header("Model Selection")
        selected_model = st.selectbox(
            "Select Model",
            options=list(MODEL_OPTIONS.keys()),
            format_func=lambda x: MODEL_OPTIONS[x]
        )
        self.embedding_service = EmbeddingService(selected_model)

    def _render_document_management(self):
        """Render document management section"""
        st.header("Document Management")
        st.subheader("Add Documents")

        col1, col2 = st.columns(2)

        with col1:
            doc_id = st.text_input("Document ID:", "doc1")
            doc_content = st.text_area(
                "Document Content:",
                "This is a sample document about machine learning."
            )

        with col2:
            if st.button("Add Document"):
                self._add_document(doc_id, doc_content)

    def _render_search_interface(self):
        """Render search interface section"""
        st.subheader("Search Documents")
        search_query = st.text_input("Enter your search query:", "machine learning")

        col1, col2 = st.columns(2)
        with col1:
            top_k = st.number_input("Number of results:", min_value=1, max_value=20, value=DEFAULT_TOP_K)

        if st.button("Search"):
            self._perform_search(search_query, top_k)

    def _render_document_display(self):
        """Render document display section"""
        st.subheader("Stored Documents")
        if st.button("Show All Documents"):
            self._show_all_documents()

    def _render_cleanup_section(self):
        """Render cleanup section"""
        st.header("Cleanup")
        if st.button("Clear All Documents", type="secondary"):
            self._clear_all_documents()

    def _add_document(self, doc_id: str, doc_content: str):
        """Add a document to the store"""
        if not doc_id or not doc_content:
            st.error("Please provide both document ID and content")
            return

        try:
            st.info(f"Using {self.embedding_service.get_model_info()}")
            embedding = self.embedding_service.generate_embedding(doc_content)

            if self.document_store.store_document(doc_id, doc_content, embedding):
                st.success(f"Successfully added document {doc_id}")
            else:
                st.error(f"Failed to add document {doc_id}")
        except Exception as e:
            st.error(f"Error adding document: {str(e)}")

    def _perform_search(self, search_query: str, top_k: int):
        """Perform document search"""
        if not search_query:
            st.error("Please enter a search query")
            return

        try:
            st.info(f"Using {self.embedding_service.get_model_info()}")
            query_embedding = self.embedding_service.generate_embedding(search_query)
            results = self.search_engine.search_documents(query_embedding, top_k)

            if results:
                st.write("Search Results:")
                for i, (doc_id, content, similarity) in enumerate(results, 1):
                    with st.expander(f"Result {i}: {doc_id} (Score: {similarity:.4f})"):
                        st.write(f"**Document ID:** {doc_id}")
                        st.write(f"**Content:** {content}")
                        st.write(f"**Similarity Score:** {similarity:.4f}")
            else:
                st.info("No documents found. Add some documents first!")

        except Exception as e:
            st.error(f"Error during search: {str(e)}")

    def _show_all_documents(self):
        """Display all stored documents"""
        docs = self.document_store.get_all_documents()
        if docs:
            for doc in docs:
                with st.expander(f"Document: {doc['id']}"):
                    st.write(f"**Content:** {doc['content']}")
                    st.write(f"**Embedding Size:** {len(doc['embedding'])}")
        else:
            st.info("No documents stored yet.")

    def _clear_all_documents(self):
        """Clear all documents from the store"""
        try:
            if self.document_store.delete_all_documents():
                st.success("Successfully cleared all documents")
            else:
                st.error("Failed to clear documents")
        except Exception as e:
            st.error(f"Error during cleanup: {str(e)}")


def main():
    """Main entry point"""
    app = DocumentSearchApp()
    app.setup_ui()


if __name__ == "__main__":
    main()