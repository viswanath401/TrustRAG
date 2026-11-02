import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="vectorstore")

collection = client.get_or_create_collection(
    name="research_papers"
)

def store_chunks(chunks):
    documents = [c["text"] for c in chunks]
    embeddings = model.encode(documents).tolist()

    metadatas = [{"page": c["page"]} for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.upsert(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    return len(chunks)

def search_chunks(query, top_k=5):
    embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=embedding,
        n_results=top_k
    )

    return results