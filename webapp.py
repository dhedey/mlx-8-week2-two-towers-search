import streamlit as st
import redis
import json
import torchdo
import numpy as np
from model import PooledTwoTowerModel, PooledOneTowerModel, TrainingHyperparameters, DualEncoderModel, ModelLoader
from model.common import select_device
import struct
from redis.commands.search.query import Query

# Connect to Redis (adjust host/port/db as needed)
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)

st.title('Document Search App')

# Dropdown for model selection
model_type = st.selectbox('Select Model', ['two-tower', 'one-tower'])

# Search input
search_query = st.text_input('Enter your search query:')

# Helper to load model
@st.cache_resource
def load_model(model_type):
    device = select_device()
    if model_type == 'two-tower':
        model_name = 'fixed-boosted-word2vec-linear'  # adjust as needed
        model = PooledTwoTowerModel.load_for_evaluation(model_name, device)
    else:
        model_name = 'learned-boosted-mini-lm-linear'  # adjust as needed
        model = PooledOneTowerModel.load_for_evaluation(model_name, device)
    return model, device

def to_float32_bytes(vec):
    arr = np.array(vec, dtype=np.float32)
    return arr.tobytes()

# Search button
if st.button('Search'):
    if not search_query:
        st.warning('Please enter a search query.')
    else:
        model, device = load_model(model_type)
        tokenized_query = model.tokenize_query(search_query)
        query_embedding = model.embed_tokenized_queries([tokenized_query]).detach().cpu().numpy()[0]
        st.write(f"Query embedding shape: {query_embedding.shape}")
        # Convert embedding to float32 bytes for Redis vector search
        query_vec_bytes = to_float32_bytes(query_embedding)
        k = 10
        base_query = f"*=>[KNN {k} @vector-two-tower-boosted-word2vec-linear $vec as score]"
        query_params = {"vec": query_vec_bytes}
        st.write(f"Base query: {base_query}")
        st.write(f"Query param keys: {list(query_params.keys())}")
        st.write(f"Query vector length: {len(query_vec_bytes)} bytes")
        # Use Redis FT.SEARCH with KNN vector search
        try:
            q = Query(base_query)
            q = q.sort_by("score")
            q = q.return_fields("url", "passage", "score")
            q = q.paging(0, k)
            q = q.dialect(2)
            res = r.ft('doc_idx').search(q, query_params)
            if hasattr(res, 'docs') and res.docs:
                st.success(f'Found {len(res.docs)} result(s):')
                for doc in res.docs:
                    # doc fields are bytes, decode as needed
                    score = getattr(doc, 'score', None)
                    url = getattr(doc, 'url', b'').decode() if isinstance(getattr(doc, 'url', b''), bytes) else getattr(doc, 'url', '')
                    passage = getattr(doc, 'passage', b'').decode(errors='ignore') if isinstance(getattr(doc, 'passage', b''), bytes) else getattr(doc, 'passage', '')
                    st.write(f"**Score:** {score}")
                    st.write(f"**URL:** {url}")
                    st.write(f"**Passage:** {passage}")
            else:
                st.info('No results found.')
        except Exception as e:
            st.error(f"Redis search error: {e}")
