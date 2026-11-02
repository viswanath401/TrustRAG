from app.pdf_processor import extract_text
from app.chunker import create_chunks
from app.hybrid_search import HybridSearch

pages = extract_text("data/research.pdf")
chunks = create_chunks(pages)

hybrid = HybridSearch(chunks)

query = input("Enter your research question: ")

results = hybrid.search(query)

print("\nHYBRID RETRIEVAL RESULTS\n")

for i, result in enumerate(results):
    print(f"\nResult {i+1}")
    print("Page:", result["page"])
    print("RRF Score:", result["score"])
    print(result["text"][:500])