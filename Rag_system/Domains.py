import re

from Rag_system.Ollama import ollama_generate

ALLOWED_DOMAINS = {"medical", "travel", "programming", "legal", "general"}
DOMAIN_SITES = {
    "medical": [
        "nih.gov", "ncbi.nlm.nih.gov", "who.int", "cdc.gov",
        "mayoclinic.org", "nhs.uk", "cochranelibrary.com"
    ],
    "travel": [
        "lonelyplanet.com", "wikivoyage.org", "swissinfo.ch",
        "sbb.ch", "sncf-connect.com", "thetrainline.com"
    ],
    "programming": [
        "docs.python.org", "learn.microsoft.com", "kubernetes.io",
        "pytorch.org", "tensorflow.org", "stackoverflow.com", "github.com"
    ],
    "legal": [
        "admin.ch", "europa.eu", "law.cornell.edu", "legifrance.gouv.fr"
    ],
    "general": []
}

DOMAIN_PATTERNS = {
    "medical": r"\b(symptom|diagnos|treat|therapy|dose|side effect|contraindicat|disease|illness|pain|fever|infection|medicine|drug|clinical|trial|ecg|cardio|blood|cancer)\b",
    "travel": r"\b(travel|trip|itinerary|visa|airport|train|hotel|budget|route|weather|best time|tickets)\b",
    "programming": r"\b(python|java|c\+\+|docker|kubernetes|api|sql|linux|azure|aws|ci/cd|git|debug|error|exception)\b",
    "legal": r"\b(contract|law|regulation|compliance|gdpr|terms|liability|court|legal)\b",
}

def detect_domain(query: str) -> str:
    q = query.lower()

    for domain, pattern in DOMAIN_PATTERNS.items():
        if re.search(pattern, q):
            if domain == "medical":
                print("⚠️ Warning: You should consult a healthcare professional. If you are experiencing a medical emergency, call your local emergency number immediately.")
            return domain

    prompt = f"""Classify the user's query into exactly one label:
        medical, travel, programming, legal, general

        Return ONLY the label.

        Query: {query}
        Label:
    """

    label = ollama_generate(prompt, temperature=0.0, timeout=40)
    label = (label or "").lower().replace(".", "").replace(":", "").split()[0]
    domain = label if label in ALLOWED_DOMAINS else "general"

    if domain == "medical":
        print("⚠️ Warning: You should consult a healthcare professional. If you are experiencing a medical emergency, call your local emergency number immediately.")
    return domain