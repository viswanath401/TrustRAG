from app.pdf_processor import extract_text
from app.chunker import create_chunks
from app.hybrid_search import HybridSearch
from app.reranker import rerank_results
from app.llm import generate_answer
from app.citation import verify_citations
from app.confidence import calculate_confidence

pages = extract_text("data/research.pdf")
chunks = create_chunks(pages)

query = input("Ask TrustRAG: ")

hybrid = HybridSearch(chunks)
results = hybrid.search(query, top_k=10)

ranked = rerank_results(query, results, top_k=5)

answer = generate_answer(query, ranked)

citation_report = verify_citations(answer, ranked)

confidence = calculate_confidence(
    ranked,
    citation_report
)

print("\nANSWER:\n")
print(answer)

print("\nCITATION REPORT:\n")
print(citation_report)

print("\nCONFIDENCE REPORT:\n")
print(confidence)