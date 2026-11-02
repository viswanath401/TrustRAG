from app.pdf_processor import extract_text
from app.chunker import create_chunks
from app.hybrid_search import HybridSearch
from app.reranker import rerank_results

pages = extract_text("data/research.pdf")
chunks = create_chunks(pages)

hybrid = HybridSearch(chunks)

query = input("Enter your research question: ")

results = hybrid.search(query, top_k=10)

ranked = rerank_results(query, results)

print("\nRERANKED RESULTS\n")

for i, result in enumerate(ranked):
    print(f"\nResult {i+1}")
    print("Page:", result["page"])
    print("Rerank Score:", round(result["rerank_score"], 4))
    print(result["text"][:500])