import re

def verify_citations(answer, results):

    available_pages = {
        result["page"] for result in results
    }

    cited_pages = re.findall(
        r"(?:Page|p\.?)\s*(\d+)",
        answer,
        re.IGNORECASE
    )

    verified = []
    invalid = []

    for page in cited_pages:
        page_number = int(page)

        if page_number in available_pages:
            verified.append(page_number)
        else:
            invalid.append(page_number)

    return {
        "available_pages": sorted(available_pages),
        "verified_pages": sorted(set(verified)),
        "invalid_pages": sorted(set(invalid)),
        "citation_status": (
            "Valid page references found"
            if cited_pages and not invalid
            else "Missing or invalid page references"
        )
    }