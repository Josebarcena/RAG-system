import os, json, faiss
import numpy as np


INDEX_PATH = "rag_index.faiss"
META_PATH = "rag_index_meta.json"

def add_docs_to_faiss_index(index, docs_meta: list[dict], new_docs: list[dict], embedder):
    """
    Añade docs al índice FAISS y a los metadatos, sin reconstruir.
    Devuelve (index, docs_meta_actualizados)
    """
    if index is None:
        test = embedder.encode(["test"], show_progress_bar=False)
        dim = int(np.array(test, dtype="float32").shape[1])
        index = faiss.IndexFlatIP(dim)

    existing = set(_doc_key(d) for d in docs_meta)
    filtered = []
    for d in new_docs:
        if not isinstance(d, dict):
            continue
        text = (d.get("text") or "").strip()
        source = (d.get("source") or "").strip()
        if not text or not source:
            continue
        k = _doc_key(d)
        if k in existing:
            continue
        existing.add(k)
        filtered.append(d)

    if not filtered:
        return index, docs_meta

    # 2) embeddings
    texts = [d["text"] for d in filtered]
    embs = embedder.encode(texts, batch_size=32, show_progress_bar=False)
    embs = np.array(embs, dtype="float32")
    embs = _normalize(embs)

    index.add(embs)
    docs_meta.extend(filtered)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(docs_meta, f, ensure_ascii=False)

    return index, docs_meta

def _doc_key(d: dict) -> str:
    # clave estable para deduplicar (source + inicio del texto)
    src = (d.get("source") or "").strip()
    txt = (d.get("text") or "").strip()
    return (src + "||" + txt[:200]).lower()

def search_faiss(query: str, embedder, index, docs_meta: list[dict], top_k: int = 20) -> list[dict]:
    if index is None or not docs_meta:
        return []

    q = embedder.encode([query], show_progress_bar=False)
    q = np.array(q, dtype="float32")
    q = _normalize(q)

    scores, idxs = index.search(q, top_k)
    idxs = idxs[0].tolist()

    out = []
    for i in idxs:
        if i == -1:
            continue
        out.append(docs_meta[i])
    return out

def build_faiss_index(docs: list[dict], embedder, index_path=INDEX_PATH, meta_path=META_PATH):
    """
    docs: list[dict] con al menos {"text":..., "source":...}
    embedder: SentenceTransformer
    """
    if faiss is None:
        raise RuntimeError("faiss no está instalado. pip install faiss-cpu")

    texts = [d["text"] for d in docs]
    embs = embedder.encode(texts, batch_size=32, show_progress_bar=True)
    embs = np.array(embs, dtype="float32")
    embs = _normalize(embs)

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine sim si normalizas
    index.add(embs)

    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)

def load_faiss_index(index_path=INDEX_PATH, meta_path=META_PATH):
    if faiss is None:
        raise RuntimeError("faiss no está instalado. pip install faiss-cpu")

    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        return None, []

    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    return index, docs

def search_faiss(query: str, embedder, index, docs: list[dict], top_k: int = 10) -> list[dict]:
    if index is None or not docs:
        return []

    q = embedder.encode([query], show_progress_bar=False)
    q = np.array(q, dtype="float32")
    q = _normalize(q)

    scores, idxs = index.search(q, top_k)
    idxs = idxs[0].tolist()

    results = []
    for i in idxs:
        if i == -1:
            continue
        results.append(docs[i])
    return results

def _normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype("float32")
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v / norms
