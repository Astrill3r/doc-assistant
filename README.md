# Doc-Assistant

Учебный проект по дисциплине «Современные технологии разработки программных систем искусственного интеллекта».

**Студент:** Гайнанов Алексей Артурович
**Группа:** МП-301
**Год:** 2026

## Стек проекта
- **Данные:** Lenta.ru (HuggingFace `zloelias/lenta-ru`), очистка через Pandas.
- **RAG (Поиск):** FAISS + `sentence-transformers` (`DeepPavlov/rubert-base-cased-sentence`).
- **LLM (Генерация):** Ollama (`qwen2.5:3b`).
- **API:** FastAPI (эндпоинт `/ask`).
- **AI-Агент:** Самописный ReAct-агент (вызов инструментов `search`, `calc`, `get_date` через JSON).
- **MLOps:** MLflow (трекинг экспериментов классического ML).
- **Развертывание:** Docker, docker-compose.
- **Данные и версионирование:** DVC (пайплайн очистки `clean_data`).

## Как запустить
1. Установить Python 3.11, Ollama, Docker Desktop.
2. Скачать модель: `ollama pull qwen2.5:3b`
3. Установить зависимости: `pip install -r requirements.txt`
4. Воспроизвести пайплайн данных: `dvc repro`
5. Построить векторный индекс: `python src/rag/build_index.py`
6. Запустить сервис: `docker-compose up --build`
7. Открыть Swagger UI для тестирования API: `http://127.0.0.1:8001/docs`

## Запуск AI-агента (локально)
```bash
python -m src.agent.manual_agent