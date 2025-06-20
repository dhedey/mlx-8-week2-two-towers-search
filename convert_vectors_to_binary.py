import redis
import json
import numpy as np

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)

# The embedding field to convert
VECTOR_FIELD = 'vector-two-tower-boosted-word2vec-linear'

# Get all document keys
keys = r.keys('doc:*')
print(f"Found {len(keys)} document keys.")

for key in keys:
    doc = r.hgetall(key)
    if VECTOR_FIELD.encode() in doc:
        # Parse the JSON string to a numpy array
        vec = np.array(json.loads(doc[VECTOR_FIELD.encode()].decode()), dtype=np.float32)
        # Store as raw float32 bytes
        r.hset(key, VECTOR_FIELD, vec.tobytes())
        print(f"Updated {key.decode()} with binary vector.")

print("Done. All vectors are now stored as float32 binary blobs.")
