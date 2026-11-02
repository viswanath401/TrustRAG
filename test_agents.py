from app.pdf_processor import extract_text
from app.chunker import create_chunks
from app.agents import research_graph

pages = extract_text("data/research.pdf")
chunks = create_chunks(pages)

question = input("Ask TrustRAG: ")

result = research_graph.invoke({
    "question": question,
    "chunks": chunks,
    "results": [],
    "answer": "",
    "verification": ""
})

print("\n========== TRUSTRAG ANSWER ==========\n")
print(result["answer"])

print("\n========== VERIFICATION REPORT ==========\n")
print(result["verification"])