import faiss
import numpy as np
from embeddings.embedder import get_embedding

# Create FAISS index
def create_index(texts):
    embeddings = get_embedding(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings
