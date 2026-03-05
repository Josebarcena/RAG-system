import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"

def ollama_generate(prompt: str,model: str = OLLAMA_MODEL,temperature: float = 0.0,timeout: int = 120,num_predict: int = 3000) -> str:
    try:

        r = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
                "num_predict": num_predict
            },
            timeout=timeout,
        )
        r.raise_for_status()
        return (r.json().get("response") or "").strip()
    except Exception as e:
        print(f"Ollama error: {e}")
        return ""

