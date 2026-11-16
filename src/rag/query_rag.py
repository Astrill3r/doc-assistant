import os
import pickle
from pathlib import Path
import faiss
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = (BASE_DIR / ".." / ".." / "data" / "vector_db").resolve()
FAISS_INDEX_PATH = DB_DIR / "faiss.index"
DOCS_PICKLE_PATH = DB_DIR / "documents_mapping.pkl"

def load_rag_assets():
    """Загружает FAISS индекс и текстовую базу в память один раз при старте сервера."""
    if not FAISS_INDEX_PATH.exists() or not DOCS_PICKLE_PATH.exists():
        raise FileNotFoundError("Ошибка: Файлы базы FAISS не найдены. Сначала запустите build_index.py")

    from sentence_transformers import SentenceTransformer
    print("Загрузка индекса FAISS и модели эмбеддингов...")

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(DOCS_PICKLE_PATH, "rb") as f:
        db_data = pickle.load(f)
    embedder = SentenceTransformer("DeepPavlov/rubert-base-cased-sentence")

    return {"index": index, "db_data": db_data, "embedder": embedder}

def rag_answer(question: str, state: dict, k: int = 3):
    """Ищет контекст в FAISS и генерирует ответ через API Ollama."""
    query_vector = state["embedder"].encode([question], convert_to_numpy=True)
    faiss.normalize_L2(query_vector)

    scores, indices = state["index"].search(query_vector, k)

    search_results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        search_results.append({
            "text": state["db_data"]["chunks"][idx],
            "distance": float(score),
            "meta": state["db_data"]["metadatas"][idx]
        })

    context = "\n---\n".join([res["text"] for res in search_results])

    system_prompt = (
        "Ты — полезный ассистент. Твоя задача — ответить на вопрос, используя ТОЛЬКО предоставленный контекст.\n"
        "Если в контексте есть ответ, дай его. Если ответа нет, напиши: «В документах не найдено».\n"
        "Контекст:\n"
    )
    user_prompt = f"ФРАГМЕНТЫ ТЕКСТА:\n{context}\n\nВОПРОС: {question}\nОТВЕТ:"

    ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/v1/chat/completions")
    payload = {
        "model": "qwen2.5:3b",
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    response = requests.post(ollama_url, json=payload, timeout=180)
    response.raise_for_status()

    llm_response = response.json()["choices"][0]["message"]["content"].strip()
    return llm_response, search_results