import redis
from redis.commands.search.query import Query
import numpy as np
import base64

print("redis-py version:", redis.__version__)

r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)

# 64-dim zero vector, base64-encoded
vec = base64.b64encode(np.zeros(64, dtype=np.float32).tobytes())
base_query = "*=>[KNN 1 @vector-two-tower-boosted-word2vec-linear $vec as score]"
params = {"vec": vec}
q = Query(base_query).sort_by("score").return_fields("url", "passage", "score").paging(0, 1).dialect(2)

print("Base query:", base_query)
print("Params:", {k: f"{len(v)} bytes" for k, v in params.items()})

try:
    res = r.ft('doc_idx').search(q, params)
    print("RESULTS:", res.docs)
except Exception as e:
    print("ERROR:", e)