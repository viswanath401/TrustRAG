from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from app.hybrid_search import HybridSearch
from app.reranker import rerank_results
from app.llm import generate_answer


class ResearchState(TypedDict):
    question: str
    chunks: list
    results: list
    answer: str
    verification: str


def query_understanding(state):
    question = state["question"].strip()

    return {"question": question}


def retrieval_agent(state):
    hybrid = HybridSearch(state["chunks"])

    results = hybrid.search(state["question"], top_k=10)

    ranked = rerank_results(
        state["question"],
        results,
        top_k=5
    )

    return {"results": ranked}


def answer_agent(state):
    answer = generate_answer(
        state["question"],
        state["results"]
    )

    return {"answer": answer}


def verification_agent(state):
    evidence = "\n".join(
        result["text"] for result in state["results"]
    )

    prompt = f"""
    You are an evidence verification agent.

    Compare the answer against the provided research evidence.

    Identify unsupported factual claims.
    Do not assume that a citation is valid merely because
    a page number is present.

    Research Evidence:
    {evidence}

    Generated Answer:
    {state["answer"]}

    Return:
    1. Supported claims
    2. Unsupported claims
    3. Overall verification status
    """

    from app.llm import client

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_completion_tokens=700
    )

    return {
        "verification": response.choices[0].message.content
    }


graph = StateGraph(ResearchState)

graph.add_node("query_understanding", query_understanding)
graph.add_node("retrieval", retrieval_agent)
graph.add_node("answer_generation", answer_agent)
graph.add_node("verification", verification_agent)

graph.add_edge(START, "query_understanding")
graph.add_edge("query_understanding", "retrieval")
graph.add_edge("retrieval", "answer_generation")
graph.add_edge("answer_generation", "verification")
graph.add_edge("verification", END)

research_graph = graph.compile()