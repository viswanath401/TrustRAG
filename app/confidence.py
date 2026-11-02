def calculate_confidence(results, citation_report):

    if not results:
        return {
            "confidence_score": 0,
            "level": "Low"
        }

    scores = [
        result["rerank_score"]
        for result in results
    ]

    best_score = max(scores)

    relevance = max(
        0,
        min(100, (best_score / (best_score + 5)) * 100)
    )

    evidence = min(100, len(results) * 20)

    citation = (
        100 if citation_report["citation_status"]
        == "Valid page references found" else 30
    )

    confidence = (
        relevance * 0.5 +
        evidence * 0.2 +
        citation * 0.3
    )

    level = (
        "High" if confidence >= 75
        else "Medium" if confidence >= 50
        else "Low"
    )

    return {
        "confidence_score": round(confidence, 2),
        "level": level,
        "relevance_score": round(relevance, 2),
        "evidence_score": evidence,
        "citation_score": citation
    }