import re

from Rag_system.Ollama import ollama_generate
from Rag_system.Retrieval import index_local_documents, retrieve_relevant_documents

# generate answer

def main():
    index_local_documents()

    while True:
        query = input("Enter your question (or 'exit' to quit): ")
        if query.lower() in ['exit', 'quit']:
            break
        answer = generate_answer(query)
        print(f"Answer: {answer}\n")

def generate_answer(query: str)-> str:
    relevant_docs = retrieve_relevant_documents(query, top_k=4)
    if not relevant_docs:
        return "I don't know\nSOURCES: []"
    
    context = "\n\n---\n\n".join(
        f"[{i}] SOURCE: {d['source']}\n{d['text']}"
        for i, d in enumerate(relevant_docs, 1)
    )
    
    prompt = f"""You are a question answering assistant.Provide a clear explanation in 4–6 sentences.
        Rules:
            1) Use ONLY information that appears explicitly in the context. Do NOT add general knowledge.
            2) You MUST cite sources using the context block numbers like [1], [2].
            3) Every sentence with factual information MUST end with at least one citation, e.g. "... [1]".
            4) Do not change numbers, dosages, or frequencies; copy them exactly from the context.
            5) If you cannot answer using the context WITH citations, reply exactly:
            I don't know

            Write the answer in this structure WITHOUT writing section titles:
            - First write a short explanation in 2–3 sentences.
            - Then write 3–6 bullet points explaining key details.
            - Then write 2–3 bullet points with practical next steps.

            Context:
            {context}

            Question:
            {query}
    """

    answer = ollama_generate(prompt, temperature=0.0, timeout=180, num_predict=300).strip()

    cited_nums = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]
    cited_nums = [n for n in cited_nums if 1 <= n <= len(relevant_docs)]
    if not cited_nums:
        return "I don't know\nSOURCES: []"

    sources = []
    seen = set()
    for n in cited_nums:
        src = relevant_docs[n - 1]["source"]
        if src not in seen:
            seen.add(src)
            sources.append(src)

    answer = re.split(r"\n\s*SOURCES\s*:", answer, maxsplit=1)[0].strip()
    answer += "\n\nSOURCES: [" + ", ".join(sources) + "]"

    return answer

if __name__ == "__main__":
    main()
