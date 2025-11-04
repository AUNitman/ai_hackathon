import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity
import json
from typing import List, Tuple

# Загрузка переменных окружения
load_dotenv()
LLM_API_KEY = os.getenv("LLM_API_KEY")
EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")

# Конфигурация
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "openrouter/mistralai/mistral-small-3.2-24b-instruct"
TOP_K_DOCUMENTS = 3  # Количество релевантных документов для контекста
CACHE_EMBEDDINGS = True  # Кэшировать эмбеддинги для экономии

class DocumentRetriever:
    """Класс для работы с базой знаний"""
    def __init__(self, embedder_api_key: str, embedding_model: str):
        self.client = OpenAI(
            base_url="https://ai-for-finance-hack.up.railway.app/",
            api_key=embedder_api_key
        )
        self.embedding_model = embedding_model
        self.documents = None
        self.doc_embeddings = None
        
    def load_documents(self, filepath: str):
        """Загрузка документов"""
        self.documents = pd.read_csv(filepath)
        print(f"Загружено: {len(self.documents)} документов")
        
    def get_embedding(self, text: str) -> List[float]:
        """Получение эмбеддинга для текста"""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def create_embeddings(self, cache_file: str = 'doc_embeddings.json'):
        """Создание эмбеддингов для всех документов с кэшированием"""
        if CACHE_EMBEDDINGS and os.path.exists(cache_file):
            print("Загрузка эмбеддингов из кэша...")
            with open(cache_file, 'r', encoding='utf-8') as f:
                embeddings_data = json.load(f)
                self.doc_embeddings = np.array(embeddings_data['embeddings'])
            print("Эмбеддинги загружены из кэша")
            return
        
        print("Создание эмбеддингов для документов...")
        embeddings = []
        for text in tqdm(self.documents['text'], desc="Эмбеддинг документов"):
            # Обрезаем очень длинные тексты для экономии
            text_truncated = text[:8000] if len(text) > 8000 else text
            embedding = self.get_embedding(text_truncated)
            embeddings.append(embedding)
        
        self.doc_embeddings = np.array(embeddings)
        
        if CACHE_EMBEDDINGS:
            embeddings_data = {
                'embeddings': self.doc_embeddings.tolist(),
                'model': self.embedding_model,
                'num_documents': len(self.documents)
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(embeddings_data, f)
            print("Эмбеддинги сохранены в кэш")
            
    def retrieve_relevant_docs(self, question: str, top_k: int = TOP_K_DOCUMENTS) -> List[Tuple[int, str, float]]:
        """Поиск top-k наиболее релевантных документов"""
        question_embedding = np.array([self.get_embedding(question)])
        
        # Вычисляем косинусное сходство
        similarities = cosine_similarity(question_embedding, self.doc_embeddings)[0]
        
        # Получаем индексы top-k документов
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Формируем результат: (id, text, similarity_score)
        results = [
            (
                self.documents.iloc[idx]['id'],
                self.documents.iloc[idx]['text'],
                similarities[idx]
            )
            for idx in top_indices
        ]
        
        return results
    
class RAGAnswerGenerator:
    """Класс для генерации ответов с использованием контекста"""
    
    def __init__(self, llm_api_key: str, llm_model: str):
        self.client = OpenAI(
            base_url="https://ai-for-finance-hack.up.railway.app/",
            api_key=llm_api_key,
        )
        self.llm_model = llm_model
    
    def create_prompt(self, question: str, context_docs: List[Tuple[int, str, float]]) -> str:
        """Создание промпта с контекстом"""
        context_parts = []
        for idx, (doc_id, text, score) in enumerate(context_docs, 1):
            context_parts.append(f"### Документ {idx} (релевантность: {score:.3f}):\n{text}\n")
        
        context = "\n".join(context_parts)
        
        prompt = f"""Ты - AI-ассистент по финансовым вопросам. Используй предоставленные документы для ответа на вопрос пользователя.

КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:
{context}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{question}

ИНСТРУКЦИИ:
1. Отвечай на основе предоставленных документов
2. Если информации недостаточно, так и скажи
3. Давай конкретный и структурированный ответ
4. Используй профессиональную терминологию
5. Если в документах есть примеры или списки - используй их

ОТВЕТ:"""
        
        return prompt
    
    def generate_answer(self, question: str, context_docs: List[Tuple[int, str, float]]) -> str:
        """Генерация ответа с контекстом"""
        prompt = self.create_prompt(question, context_docs)
        
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "Ты - профессиональный AI-ассистент по финансовым вопросам. Отвечай точно, структурированно и на основе предоставленного контекста."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,  # Низкая температура для более факуальных ответов
            max_tokens=1000
        )
        
        return response.choices[0].message.content


def evaluate_answers(questions_df: pd.DataFrame, retriever: DocumentRetriever) -> pd.DataFrame:
    """Добавление метрик для оценки качества"""
    
    metrics = []
    for idx, row in questions_df.iterrows():
        question = row['Вопрос']
        answer = row['Ответы на вопрос']
        
        # Получаем релевантные документы для анализа
        relevant_docs = retriever.retrieve_relevant_docs(question, top_k=1)
        top_similarity = relevant_docs[0][2] if relevant_docs else 0
        
        # Простые метрики
        answer_length = len(answer.split())
        has_structure = any(marker in answer for marker in ['1.', '2.', '-', '•', '*'])
        
        metrics.append({
            'question_id': row['id'],
            'answer_length': answer_length,
            'has_structure': has_structure,
            'top_doc_similarity': top_similarity
        })
    
    metrics_df = pd.DataFrame(metrics)
    return metrics_df


if __name__ == "__main__":
    print("=" * 50)
    print("Запуск RAG-системы для финансового AI-ассистента")
    print("=" * 50)
    
    # Инициализация компонентов
    retriever = DocumentRetriever(
        embedder_api_key=EMBEDDER_API_KEY,
        embedding_model=EMBEDDING_MODEL
    )
    
    generator = RAGAnswerGenerator(
        llm_api_key=LLM_API_KEY,
        llm_model=LLM_MODEL
    )
    
    # Загрузка и подготовка базы знаний
    print("\n1. Загрузка базы знаний...")
    retriever.load_documents('./train_data.csv')
    
    print("\n2. Создание эмбеддингов...")
    retriever.create_embeddings()
    
    # Загрузка вопросов
    print("\n3. Загрузка вопросов...")
    questions = pd.read_csv('./questions.csv')
    questions_list = questions['Вопрос'].tolist()
    
    # Генерация ответов
    print("\n4. Генерация ответов с использованием RAG...")
    answer_list = []
    relevant_docs_info = []
    
    for current_question in tqdm(questions_list, desc="Обработка вопросов"):
        # Поиск релевантных документов
        relevant_docs = retriever.retrieve_relevant_docs(current_question, top_k=TOP_K_DOCUMENTS)
        
        # Генерация ответа с контекстом
        answer = generator.generate_answer(current_question, relevant_docs)
        
        answer_list.append(answer)
        relevant_docs_info.append({
            'top_similarity': relevant_docs[0][2],
            'used_docs': [doc[0] for doc in relevant_docs]
        })
    
    # Добавление ответов
    questions['Ответы на вопрос'] = answer_list
    
    # Оценка качества
    print("\n5. Оценка качества ответов...")
    metrics_df = evaluate_answers(questions, retriever)
    
    # Вывод статистики
    print("\n" + "=" * 50)
    print("СТАТИСТИКА:")
    print("=" * 50)
    print(f"Средняя длина ответа: {metrics_df['answer_length'].mean():.1f} слов")
    print(f"Ответов со структурой: {metrics_df['has_structure'].sum()} / {len(metrics_df)}")
    print(f"Средняя релевантность top документа: {metrics_df['top_doc_similarity'].mean():.3f}")
    print(f"Минимальная релевантность: {metrics_df['top_doc_similarity'].min():.3f}")
    
    # Сохранение результатов
    print("\n6. Сохранение результатов...")
    questions.to_csv('submission.csv', index=False)
    metrics_df.to_csv('evaluation_metrics.csv', index=False)
    
    print("\n" + "=" * 50)
    print("✓ Готово! Файлы сохранены:")
    print("  - submission.csv (основной файл с ответами)")
    print("  - evaluation_metrics.csv (метрики качества)")
    print("  - doc_embeddings.json (кэш эмбеддингов)")
    print("=" * 50)