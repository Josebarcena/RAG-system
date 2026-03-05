from io import BytesIO
import re, requests
import numpy as np

from ddgs import DDGS
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from bs4 import BeautifulSoup
from pypdf import PdfReader

from Rag_system.Files import save_document
from Rag_system.Domains import DOMAIN_SITES, detect_domain
from concurrent.futures import ThreadPoolExecutor, as_completed
from Rag_system.Files import load_local_documents
from Rag_system.Vector import build_faiss_index, load_faiss_index, add_docs_to_faiss_index, search_faiss


MODEL = SentenceTransformer("all-MiniLM-L6-v2")
RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
FAISS_INDEX, FAISS_DOCS = load_faiss_index()

def index_local_documents()-> None:
    docs = []
    for doc in load_local_documents():
        for c in chunk_text(doc["text"]):
            docs.append({
                "text": c,
                "source": doc["source"],
                "title": doc.get("title", ""),
                "domain": doc.get("domain", "local"),
            })

    build_faiss_index(docs, MODEL)

def retrieve_relevant_documents(query: str, top_k: int = 8) -> list[dict]:
    global FAISS_INDEX, FAISS_DOCS
    domain = detect_domain(query)
    candidates = search_faiss(query, MODEL, FAISS_INDEX, FAISS_DOCS, top_k=60)
    candidates = limit_chunks_per_source(candidates, max_per_source=1)

    local_top = rank_documents(query, candidates, top_k=top_k)
    score = best_rerank_score(query, local_top)
    needs_web = (len(local_top) < top_k) or (score < 2.0)

    if needs_web:
        web_results = search_web(query, max_results=10, domain=domain)

        web_chunks = []
        workers = min(max(2, int(len(web_results) * 0.2)), 6)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(process_web_result, r, query) for r in web_results]
            for fut in as_completed(futures):
                web_chunks.extend(fut.result())

        web_chunks = limit_chunks_per_source(web_chunks, max_per_source=1)

        if web_chunks:
            FAISS_INDEX, FAISS_DOCS = add_docs_to_faiss_index(
                FAISS_INDEX, FAISS_DOCS, web_chunks, MODEL
            )

        candidates2 = search_faiss(query, MODEL, FAISS_INDEX, FAISS_DOCS, top_k=80)
        candidates2 = limit_chunks_per_source(candidates2, max_per_source=1)
        return rank_documents(query, candidates2, top_k=top_k)

    return local_top



def limit_chunks_per_source(docs, max_per_source=2):
    counts = {}
    out = []

    for d in docs:
        src = d["source"]
        counts[src] = counts.get(src, 0)

        if counts[src] < max_per_source:
            out.append(d)
            counts[src] += 1

    return out

def chunk_text(text: str, chunk_size: int = 350, overlap: int = 80) -> list[str]:

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+chunk_size]
        if len(chunk) > 50:
            chunks.append(chunk)
        i += max(1, chunk_size - overlap)
    return chunks

def rank_documents(query: str, docs: list[dict], top_k: int = 3) -> list[dict]:
    if not docs:
        return []

    texts = [d["text"] for d in docs]

    # BM25
    tokenized_docs = [t.split() for t in texts]
    bm25 = BM25Okapi(tokenized_docs)
    bm25_scores = bm25.get_scores(query.split())

    # Embeddings
    query_emb = MODEL.encode(query, show_progress_bar=False)
    doc_embs = MODEL.encode(texts, batch_size=32, show_progress_bar=False)

    query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    doc_embs = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-9)

    sim_scores = doc_embs @ query_emb

    bm25_scores = bm25_scores / (np.max(bm25_scores) + 1e-9)
    sim_scores = sim_scores / (np.max(sim_scores) + 1e-9)

    hybrid_scores = 0.5 * bm25_scores + 0.5 * sim_scores

    ranked = [doc for doc, _ in sorted(zip(docs, hybrid_scores), key=lambda x: x[1], reverse=True)]
    candidates = ranked[: min(10, len(ranked))]

    pairs = [[query, d["text"]] for d in candidates]
    rerank_scores = RERANKER.predict(pairs)

    final_docs = [doc for doc, _ in sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)]

    return final_docs[:top_k]

def best_rerank_score(query: str, docs: list[dict]) -> float:
    if not docs:
        return -1e9
    texts = []
    for d in docs[: min(5, len(docs))]:
        t = (d.get("text") or "").strip()
        if t:
            texts.append(t)

    if not texts:
        return -1e9

    pairs = [[query, t] for t in texts]
    scores = RERANKER.predict(pairs)
    return float(np.max(scores)) if len(scores) else -1e9

def search_web(query, max_results=15, domain:str = None):
    def build_search_query(query: str) -> tuple[str, str]:
        sites = DOMAIN_SITES.get(domain, [])
        if not sites:
            return query, domain

        # (site:a OR site:b OR site:c) query
        sites_expr = " OR ".join([f"site:{s}" for s in sites[:6]])
        return f"({sites_expr}) {query}", domain
    try:
        routed_query, domain = build_search_query(query)
        results = []

        with DDGS() as ddgs:
            for r in ddgs.text(routed_query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "href": r.get("href", ""),
                    "body": r.get("body", "") or "",
                    "domain": domain
                })
        return results
    except Exception as e:
        print(f"Search error: {e}")
        return []

def process_web_result(r: dict, query: str) -> list[dict]:
    href = (r.get("href") or "").strip()
    domain = r.get("domain", "general")

    snippet = (r.get("body") or "").strip()
    title = (r.get("title") or "").strip()

    text = ""
    if href:
        try:
            text = fetch_page_text(href)
        except Exception:
            text = ""

    if not text:
        text = f"{title}\n{href}\n\n{snippet}".strip()
        if not text:
            return []

    text = text[:20000]

    save_document(text, source=href or title or "duckduckgo", query=query, domain=domain)

    chunks = chunk_text(text)
    return [{"text": c, "source": href or title or "duckduckgo", "title": title, "domain": domain} for c in chunks]

def fetch_page_text(url: str, timeout: int = 12) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()

    ctype = (r.headers.get("Content-Type") or "").lower()

    # PDF
    if "application/pdf" in ctype or url.lower().endswith(".pdf"):
        reader = PdfReader(BytesIO(r.content))
        text = ""
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
        return re.sub(r"\s+", " ", text).strip()

    # HTML
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()

