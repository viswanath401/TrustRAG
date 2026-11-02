from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_results(query, results, top_k=5):

    pairs = [
        (query, result["text"])
        for result in results
    ]

    scores = model.predict(pairs)

    for result, score in zip(results, scores):
        result["rerank_score"] = float(score)

    ranked = sorted(
        results,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return ranked[:top_k]