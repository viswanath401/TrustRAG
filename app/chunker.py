from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []

    for page in pages:
        texts = splitter.split_text(page["text"])

        for text in texts:
            chunks.append({
                "text": text,
                "page": page["page"]
            })

    return chunks