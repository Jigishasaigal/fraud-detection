import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss

# ----------------------------------
# Paths
# ----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "documents")
INDEX_PATH = os.path.join(BASE_DIR, "index.faiss")
META_PATH = os.path.join(BASE_DIR, "index_meta.pkl")

# ----------------------------------
# Load embedding model
# ----------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
metadata = []

# ----------------------------------
# Read documents
# ----------------------------------
for filename in os.listdir(DOCS_DIR):
    file_path = os.path.join(DOCS_DIR, filename)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

        # split by blank lines → logical chunks
        chunks = [c.strip() for c in content.split("\n\n") if c.strip()]

        for chunk in chunks:
            documents.append(chunk)
            metadata.append({
                "source": filename,
                "text": chunk
            })

# ----------------------------------
# Create embeddings
# ----------------------------------
embeddings = model.encode(documents, show_progress_bar=True)

# ----------------------------------
# Build FAISS index
# ----------------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# ----------------------------------
# Save index + metadata
# ----------------------------------
faiss.write_index(index, INDEX_PATH)

with open(META_PATH, "wb") as f:
    pickle.dump(metadata, f)

print("RAG index built successfully.")
print(f"Documents indexed: {len(documents)}")
