import json
from pathlib import Path

p = Path("data/processed/concept_cache_image/CXR3378.json")
d = json.load(open(p, encoding="utf-8"))
print("nb concepts:", len(d["concepts"]))
for c in d["concepts"]:
    name = c["canonical_name"]
    conf = c["confidence"]
    dim = len(c["embedding"])
    print("  " + name.ljust(24) + " p=" + format(conf, ".3f") + "  emb_dim=" + str(dim))
edges = d["edge_weights"]
rows = len(edges)
cols = len(edges[0]) if edges else 0
print("edge shape:", rows, "x", cols)
