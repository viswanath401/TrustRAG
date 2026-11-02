from app.pdf_processor import extract_text
from app.chunker import create_chunks
from app.hybrid_search import HybridSearch
from app.reranker import rerank_results
from app.llm import generate_answer

pages = extract_text("data/research.pdf")
chunks = create_chunks(pages)

hybrid = HybridSearch(chunks)

query = input("Ask your research question: ")

results = hybrid.search(query, top_k=10)
ranked = rerank_results(query, results, top_k=5)

answer = generate_answer(query, ranked)

print("\n========== TRUST RAG ANSWER ==========\n")
print(answer)