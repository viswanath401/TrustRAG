
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    try:
        import streamlit as st
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    raise ValueError("GROQ_API_KEY not found. Configure it in .env or Streamlit Secrets.")

client = Groq(api_key=api_key)


def generate_answer(query, results):

    context = ""

    for i, result in enumerate(results):
        context += f"""
Source {i+1}
Page: {result['page']}
Content: {result['text']}
"""

    prompt = f"""
You are TrustRAG, an AI research assistant.

Answer the user's question using ONLY the research paper
context provided below.

Rules:
1. Do not invent facts.
2. If evidence is insufficient, say so.
3. Cite page numbers for factual claims.
4. Give clear and structured answers.
5. Distinguish evidence from interpretation.

Research Context:
{context}

User Question:
{query}

Provide your answer with citations.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_completion_tokens=1500
    )

    return response.choices[0].message.content
