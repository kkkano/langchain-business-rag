from __future__ import annotations

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        if not texts:
            return []
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str):
        return self.model.encode([text], normalize_embeddings=True).tolist()[0]
