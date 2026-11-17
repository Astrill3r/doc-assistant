import json
import re
import requests
from src.agent.agent_tools import TOOLS, TOOLS_DESCRIPTION

SYSTEM_PROMPT = f"""Ты — AI-агент с доступом к инструментам.
Доступные инструменты:
{TOOLS_DESCRIPTION}

Чтобы решить задачу пользователя, ты должен рассуждать пошагово и возвращать СТРОГО JSON в одном из форматов:

Формат 1 (Вызов инструмента):
{{"action": "search", "input": "запрос для поиска"}}
ИЛИ
{{"action": "calc", "input": "25 * 4"}}

Формат 2 (Финальный ответ, когда информация собрана):
{{"action": "final", "answer": "Твой финальный ответ пользователю"}}

ПРАВИЛА:
1. Не выдумывай факты. Если не знаешь — используй search.
2. Возвращай только чистый JSON, без markdown-тегов (```json) и пояснений.
"""

def ask_ollama(prompt: str, system_prompt: str = "", temperature: float = 0.1) -> str:
    url = "http://127.0.0.1:11434/v1/chat/completions"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    payload = {"model": "qwen2.5:3b", "temperature": temperature, "messages": messages}
    response = requests.post(url, json=payload, timeout=180)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

def run_agent(question: str, max_steps: int = 5) -> str:
    history = f"Вопрос пользователя: {question}\n\n"
    
    for step in range(1, max_steps + 1):
        print(f"\n--- Шаг {step} ---")
        raw_response = ask_ollama(history, system_prompt=SYSTEM_PROMPT, temperature=0.1)
        print(f"LLM ответила:\n{raw_response}")
        
        clean_json = re.sub(r'^```json\s*|\s*```$', '', raw_response.strip())
        
        try:
            decision = json.loads(clean_json)
        except json.JSONDecodeError:
            return "Ошибка: LLM вернула некорректный JSON."

        action = decision.get("action")
        
        if action == "final":
            return decision.get("answer", "Ответ не сформирован.")
        
        tool_name = action
        tool_input = decision.get("input", "")
        
        if tool_name in TOOLS:
            print(f"Вызов инструмента: {tool_name}('{tool_input}')")
            result = TOOLS[tool_name](tool_input)
            print(f"Результат: {str(result)[:150]}...")
            history += f"Шаг {step}: {tool_name}({tool_input}) -> Результат: {result}\n"
        else:
            history += f"Шаг {step}: Ошибка, неизвестный инструмент {tool_name}\n"

    return "Агент не смог решить задачу за отведенное количество шагов."

if __name__ == "__main__":
    query = "Сколько дней основного отпуска по трудовому кодексу в документах? Умножь это число на 8 рабочих часов."
    print(f"Запуск агента с вопросом: {query}")
    answer = run_agent(query)
    print(f"\nИтоговый ответ: {answer}")