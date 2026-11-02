from app.pdf_processor import extract_text
from app.chunker import create_chunks
from app.vector_store import store_chunks, search_chunks

pages = extract_text("data/research.pdf")
chunks = create_chunks(pages)

count = store_chunks(chunks)

print("Stored Chunks:", count)

query = input("\nEnter your research question: ")

results = search_chunks(query)

print("\nRelevant Research Content:\n")

for i, text in enumerate(results["documents"][0]):
    print(f"\nResult {i+1}")
    print("Page:", results["metadatas"][0][i]["page"])
    print(text)