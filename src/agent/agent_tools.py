import math
from datetime import datetime
from src.rag.query_rag import load_rag_assets

print("Загрузка векторной базы для агента...")
rag_state = load_rag_assets()

def tool_search(query: str) -> str:
    query_vector = rag_state["embedder"].encode([query], convert_to_numpy=True)
    import faiss
    faiss.normalize_L2(query_vector)
    scores, indices = rag_state["index"].search(query_vector, k=3)
    chunks = []
    for idx in indices[0]:
        if idx != -1:
            chunks.append(rag_state["db_data"]["chunks"][idx])
    return "\n---\n".join(chunks) if chunks else "Ничего не найдено."

def tool_calc(expression: str) -> str:
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Ошибка вычисления: {e}"

def tool_get_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")

TOOLS = {
    "search": tool_search,
    "calc": tool_calc,
    "get_date": tool_get_date
}

TOOLS_DESCRIPTION = """
1. search(query: str) - поиск информации в базе знаний.
2. calc(expression: str) - вычисление математических выражений (например, '10 * 8').
3. get_date() - получить текущую дату.
"""