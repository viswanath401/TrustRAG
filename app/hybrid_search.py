from app.vector_store import search_chunks
from app.bm25_search import BM25Search

class HybridSearch:

    def __init__(self, chunks):
        self.chunks = chunks
        self.bm25 = BM25Search(chunks)

    def search(self, query, top_k=5):

        dense = search_chunks(query, top_k=10)
        sparse = self.bm25.search(query, top_k=10)

        scores = {}
        documents = {}

        for rank, text in enumerate(dense["documents"][0]):
            key = text
            scores[key] = scores.get(key, 0) + 1 / (60 + rank + 1)
            documents[key] = {
                "text": text,
                "page": dense["metadatas"][0][rank]["page"]
            }

        for rank, item in enumerate(sparse):
            key = item["text"]
            scores[key] = scores.get(key, 0) + 1 / (60 + rank + 1)
            documents[key] = {
                "text": key,
                "page": item["page"]
            }

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            {
                **documents[text],
                "score": round(score, 5)
            }
            for text, score in ranked[:top_k]
        ]