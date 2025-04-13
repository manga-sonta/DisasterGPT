from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL_NAME

model = SentenceTransformer(EMBED_MODEL_NAME)

def get_embedding(texts):
    return model.encode(texts, show_progress_bar=True)
