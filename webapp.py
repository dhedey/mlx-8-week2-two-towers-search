import streamlit as st
import redis
import json

# Connect to Redis (adjust host/port/db as needed)
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

st.title('Document Search App')

# Dropdown for model selection
model_type = st.selectbox('Select Model', ['two-tower', 'one-tower'])

# Search input
search_query = st.text_input('Enter your search query:')

# Search button
if st.button('Search'):
    if not search_query:
        st.warning('Please enter a search query.')
    else:
        # Example: Assume Redis stores documents as JSON strings with keys 'doc:<id>'
        # and a sorted set 'docs' for all document IDs
        doc_ids = r.zrange('docs', 0, -1)
        results = []
        for doc_id in doc_ids:
            doc = r.get(f'doc:{doc_id}')
            if doc:
                doc_data = json.loads(doc)
                # Simple search: check if query in document text (case-insensitive)
                if search_query.lower() in doc_data.get('text', '').lower():
                    results.append(doc_data)
        if results:
            st.success(f'Found {len(results)} result(s):')
            for doc in results:
                st.write(doc)
        else:
            st.info('No results found.')
