from app.pdf_processor import extract_text
from app.chunker import create_chunks
from app.bm25_search import BM25Search

pages = extract_text("data/research.pdf")
chunks = create_chunks(pages)

bm25 = BM25Search(chunks)

query = input("Enter your research question: ")

results = bm25.search(query)

for i, result in enumerate(results):
    print(f"\nResult {i+1}")
    print("Page:", result["page"])
    print("Score:", round(result["score"], 3))
    print(result["text"][:500])