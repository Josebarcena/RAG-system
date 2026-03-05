# Domain-Aware RAG System

A Retrieval-Augmented Generation (RAG) system that combines semantic
search, web retrieval, and source-grounded answer generation.

The system retrieves relevant documents from a local knowledge base and
the web, reranks them using a cross-encoder model, and generates answers
strictly grounded in the retrieved context with explicit citations.

------------------------------------------------------------------------

## Features

-   Hybrid retrieval (BM25 + embeddings)
-   Vector search with FAISS
-   Cross-encoder reranking for high precision
-   Automatic web search fallback when local context is insufficient
-   Domain-aware query routing
-   Source-grounded answers with citations
-   Medical safety guardrails
-   Document chunking and indexing
-   Automatic knowledge base growth from web retrieval

------------------------------------------------------------------------

## Architecture

User Query\
↓\
Domain Detection\
↓\
Vector Retrieval (FAISS)\
↓\
Hybrid Ranking (BM25 + embeddings)\
↓\
Cross-Encoder Reranking\
↓\
Quality Check\
↓\
Web Search Fallback (if needed)\
↓\
Context Assembly\
↓\
LLM Answer Generation\
↓\
Source Grounding + Citations

------------------------------------------------------------------------

## Example Output

Question: what are the symptoms of Lyme disease

Answer: Lyme disease symptoms vary depending on the stage of infection.
Early symptoms may include fever, rash, fatigue, and headache.

• Fever may occur in the early stage of Lyme disease \[1\].\
• A bull's-eye rash is a common sign of infection \[1\].\
• Facial paralysis may develop if untreated \[1\].\
• Arthritis may appear in later stages of the disease \[1\].

SOURCES: https://www.cdc.gov/lyme/signs-symptoms/index.html\
https://www.nccih.nih.gov/health/Lyme-Disease

------------------------------------------------------------------------

## Project Structure

Rag-System/ │ ├── Rag_system/ │ ├── Rag.py \# main entry point │ ├──
Retrieval.py \# retrieval pipeline │ ├── Domain.py \# domain detection │
├── Files.py \# document storage │ ├── vector_store.py \# FAISS vector
database │ ├── Ollama.py \# LLM interface │ ├── docs/ \# downloaded
source documents ├── rag_index.faiss \# FAISS vector index ├──
rag_index_meta.json \# document metadata │ └── README.md

------------------------------------------------------------------------

## Installation

Install dependencies:

pip install sentence-transformers pip install faiss-cpu pip install
duckduckgo-search pip install beautifulsoup4 pip install requests pip
install rank-bm25

Install Ollama (for local LLM):

https://ollama.ai

Example model:

ollama pull llama3

------------------------------------------------------------------------

## Usage

Run the system:

python -m Rag_system.Rag

Then ask questions:

Enter your question: what are the treatments for porphyria

------------------------------------------------------------------------

## How It Works

### Retrieval Pipeline

1.  Query domain detection
2.  Vector search (FAISS)
3.  Hybrid ranking (BM25 + embeddings)
4.  Cross-encoder reranking
5.  Quality threshold evaluation
6.  Web search fallback if needed

### Answer Generation

The LLM receives only retrieved context and must:

-   cite sources using \[1\], \[2\]
-   avoid adding external knowledge
-   return "I don't know" if context is insufficient

------------------------------------------------------------------------

## Safety

Medical queries trigger a safety disclaimer:

⚠️ You should consult a healthcare professional.

The system also prevents hallucinations by requiring citations for
factual statements.

------------------------------------------------------------------------

## Technologies

-   Python
-   FAISS
-   SentenceTransformers
-   CrossEncoder reranking
-   DuckDuckGo Search
-   BeautifulSoup
-   Ollama LLMs
-   BM25 ranking

------------------------------------------------------------------------

## Future Improvements

-   Query caching
-   Domain-specific retrieval filters
-   UI interface (FastAPI / Streamlit)
-   Source reliability scoring
-   PDF parsing support
-   Distributed vector database

------------------------------------------------------------------------

## License

MIT
