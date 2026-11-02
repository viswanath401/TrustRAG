from rank_bm25 import BM25Okapi

class BM25Search:
    def __init__(self, chunks):
        self.chunks = chunks
        tokenized = [c["text"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query, top_k=5):
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return [
            {
                "text": self.chunks[i]["text"],
                "page": self.chunks[i]["page"],
                "score": float(score)
            }
            for i, score in ranked
        ]