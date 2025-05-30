# Legal Search Engine

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Opening JSON file
with open('MVA.json', encoding="utf8") as f:
    data = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

# Combine title + description for better embeddings
corpus_texts = [f"{entry['section']} {entry['title']} {entry['description']}" for entry in data]

embeddings = model.encode(corpus_texts, convert_to_numpy=True)

dimension = embeddings.shape[1]  # 384 for MiniLM
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

query = "Can I drive a scooter without a licence?"
query_embedding = model.encode([query])
k = 3
distances, indices = index.search(np.array(query_embedding), k)

# Print results
print(f"\n🔍 Query: {query}\nTop {k} relevant sections:\n")
for idx in indices[0]:
    section = data[idx]
    print(f"📘 Section {section['section']}: {section['title']}")
    print(f"{section['description']}\n")
    print("-" * 80)