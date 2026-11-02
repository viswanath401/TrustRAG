from app.pdf_processor import extract_text
from app.chunker import create_chunks

pages = extract_text("data/research.pdf")
chunks = create_chunks(pages)

print("Total Pages:", len(pages))
print("Total Chunks:", len(chunks))
print("\nFirst Chunk:\n")
print(chunks[0]["text"])
print("\nPage Number:", chunks[0]["page"])