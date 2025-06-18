import streamlit as st
import redis
import os

# Redis connection
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', 6379))

st.title("Redis Testing Interface")

# Initialize Redis connection
try:
    redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    st.success(f"Successfully connected to Redis at {redis_host}:{redis_port}")
except redis.ConnectionError as e:
    st.error(f"Failed to connect to Redis: {str(e)}")
    redis_client = None

if redis_client:
    # Test Redis Operations
    st.header("Test Redis Operations")

    # Set/Get Test
    st.subheader("Set/Get Test")
    test_key = st.text_input("Enter a key for testing:", "test_key")
    test_value = st.text_input("Enter a value for testing:", "test_value")

    if st.button("Test Set/Get"):
        try:
            redis_client.set(test_key, test_value)
            retrieved_value = redis_client.get(test_key)
            st.success(f"Successfully set and retrieved value: {retrieved_value}")
        except Exception as e:
            st.error(f"Error during Set/Get test: {str(e)}")

    # List Operations Test
    st.subheader("List Operations Test")
    list_key = st.text_input("Enter a list key:", "test_list")
    list_value = st.text_input("Enter a value to add to list:", "test_item")

    if st.button("Test List Operations"):
        try:
            redis_client.lpush(list_key, list_value)
            list_length = redis_client.llen(list_key)
            list_items = redis_client.lrange(list_key, 0, -1)
            st.success(f"List length: {list_length}")
            st.write("List items:", list_items)
        except Exception as e:
            st.error(f"Error during List operations test: {str(e)}")

    # Hash Operations Test
    st.subheader("Hash Operations Test")
    hash_key = st.text_input("Enter a hash key:", "test_hash")
    hash_field = st.text_input("Enter a hash field:", "test_field")
    hash_value = st.text_input("Enter a hash value:", "test_value")

    if st.button("Test Hash Operations"):
        try:
            redis_client.hset(hash_key, hash_field, hash_value)
            retrieved_hash = redis_client.hget(hash_key, hash_field)
            st.success(f"Successfully set and retrieved hash value: {retrieved_hash}")
        except Exception as e:
            st.error(f"Error during Hash operations test: {str(e)}")

    # Cleanup Section
    st.header("Cleanup")
    if st.button("Clear All Test Data"):
        try:
            redis_client.delete(test_key, list_key, hash_key)
            st.success("Successfully cleared test data")
        except Exception as e:
            st.error(f"Error during cleanup: {str(e)}")