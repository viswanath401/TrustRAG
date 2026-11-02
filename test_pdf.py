from app.pdf_processor import extract_text

pages = extract_text("data/research.pdf")

print("Total Pages:", len(pages))
print("\nExtracted Text:\n")
print(pages[0]["text"][:1000])