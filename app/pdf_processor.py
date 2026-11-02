import fitz

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages.append({
                "page": i + 1,
                "text": text
            })

    doc.close()
    return pages