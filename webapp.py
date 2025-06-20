import streamlit as st
import redis
import json
import torch
import numpy as np
from model.models import PooledTwoTowerModel, PooledOneTowerModel, TrainingHyperparameters
from model.models import DualEncoderModel
from model.models import ModelLoader
from model.common import select_device
import struct

# Connect to Redis (adjust host/port/db as needed)
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

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
        # Convert embedding to float32 bytes for Redis vector search
        query_vec_bytes = to_float32_bytes(query_embedding)
        # Use Redis FT.SEARCH with KNN vector search
        k = 10
        base_query = f"*=>[KNN {k} @vector-two-tower-boosted-word2vec-linear $vec as score]"
        query_params = {"vec": query_vec_bytes}
        res = r.ft('doc_idx').search(base_query, query_params=query_params)
        if res.docs:
            st.success(f'Found {len(res.docs)} result(s):')
            for doc in res.docs:
                st.write(f"**Score:** {doc.score}")
                st.write(f"**URL:** {getattr(doc, 'url', '')}")
                st.write(f"**Passage:** {getattr(doc, 'passage', '')}")
        else:
            st.info('No results found.')
