from sentence_transformers import SentenceTransformer
import torch

class EmbeddingManager:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.device = "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"Embedding model '{model_name}' loaded on {self.device}")

    def generate_embeddings(self, texts: list[str]):
        return self.model.encode(texts, convert_to_numpy=True)

    def generate_query_embedding(self, query: str):
        embedding = self.model.encode([query], convert_to_numpy=True)
        # Ensure it's 2D (1, N) for FAISS
        if len(embedding.shape) == 1:
            embedding = embedding.reshape(1, -1)
        return embedding
