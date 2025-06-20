import streamlit as st
import redis
import json
import torch
import numpy as np
from model.models import PooledTwoTowerModel, PooledOneTowerModel, TrainingHyperparameters
from model.models import DualEncoderModel
from model.models import ModelLoader
from model.common import select_device

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

# Search button
if st.button('Search'):
    if not search_query:
        st.warning('Please enter a search query.')
    else:
        model, device = load_model(model_type)
        # Tokenize and embed the query
        tokenized_query = model.tokenize_query(search_query)
        query_embedding = model.embed_tokenized_queries([tokenized_query]).detach().cpu().numpy()[0]

        # Retrieve all document IDs
        doc_ids = r.zrange('docs', 0, -1)
        results = []
        for doc_id in doc_ids:
            doc = r.get(f'doc:{doc_id}')
            doc_emb = r.get(f'doc_emb:{doc_id}')
            if doc and doc_emb:
                doc_data = json.loads(doc)
                doc_embedding = np.array(json.loads(doc_emb))
                # Compute cosine similarity
                sim = np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding) + 1e-8)
                results.append((sim, doc_data))
        if results:
            results.sort(reverse=True, key=lambda x: x[0])
            st.success(f'Found {len(results)} result(s):')
            for sim, doc in results[:10]:
                st.write(f"**Score:** {sim:.3f}")
                st.write(doc)
        else:
            st.info('No results found.')
