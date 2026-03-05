
import os,re,hashlib,json

from datetime import datetime, timezone

DOCS_DIR = "documents"

def load_local_documents() -> list[str]:
    ensure_docs_dir()
    docs = []

    for name in os.listdir(DOCS_DIR):
        if not name.lower().endswith(".txt"):
            continue

        path = os.path.join(DOCS_DIR, name)

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()

            parts = raw.split("=== CONTENT ===\n", 1)
            meta_part = parts[0] if len(parts) > 0 else ""
            content = parts[1].strip() if len(parts) > 1 else ""

            if not content:
                continue

            meta = {}
            if "=== METADATA ===" in meta_part:
                meta_lines = meta_part.splitlines()
                for i, line in enumerate(meta_lines):
                    if line.strip() == "=== METADATA ===" and i + 1 < len(meta_lines):
                        try:
                            meta = json.loads(meta_lines[i + 1])
                        except Exception:
                            meta = {}
                        break

            source = meta.get("source") or path
            domain = meta.get("domain")
            query = meta.get("query") or ""

            docs.append({
                "text": content,
                "source": source,
                "domain": domain,
                "title": os.path.basename(path),
                "query": query,
                "path": path
            })

        except Exception:
            continue

    return docs

def save_document(text: str, source: str, query: str, domain: str = "general") -> str:
    ensure_docs_dir()

    # deterministic-ish filename to avoid duplicates
    key = sha1(text[:500] + source)
    fname = f"{safe_filename(query)}__{key[:10]}.txt"
    path = os.path.join(DOCS_DIR, fname)

    if os.path.exists(path):
        return path

    header = {
        "source": source,
        "query": query,
        "domain": domain,
        "saved_at": datetime.now(timezone.utc).isoformat()
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("=== METADATA ===\n")
            f.write(json.dumps(header, ensure_ascii=False) + "\n")
            f.write("=== CONTENT ===\n")
            f.write(text.strip() + "\n")

    except Exception as e:
        print(f"Error saving document: {e}")

    return path

def ensure_docs_dir() -> None:
    os.makedirs(DOCS_DIR, exist_ok=True)

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def safe_filename(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")
    return text[:max_len] if text else "doc"

