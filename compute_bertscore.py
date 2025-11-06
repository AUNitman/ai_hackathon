import os
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Берём bert-score из пакета
from bert_score import score as bertscore_score


def find_most_relevant_docs(questions, docs_texts, docs_ids, vectorizer=None, batch_size=256):
    """Векторизуем документы (если vectorizer=None), затем для каждого вопроса находим наиболее релевантный документ.
    Возвращает списки: best_doc_id, best_doc_text, best_similarity
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
        doc_mat = vectorizer.fit_transform(docs_texts)
    else:
        doc_mat = vectorizer.transform(docs_texts)

    best_ids = []
    best_texts = []
    best_sims = []

    n = len(questions)
    for start in tqdm(range(0, n, batch_size), desc="Finding relevant docs"):
        end = min(n, start + batch_size)
        q_batch = questions[start:end]
        q_mat = vectorizer.transform(q_batch)
        sims = cosine_similarity(q_mat, doc_mat)
        # Для каждой строки найдём argmax
        idx = sims.argmax(axis=1)
        vals = sims.max(axis=1)
        for i, j in enumerate(idx):
            best_ids.append(docs_ids[j])
            best_texts.append(docs_texts[j])
            best_sims.append(float(vals[i]))

    return best_ids, best_texts, best_sims, vectorizer


if __name__ == '__main__':
    # Путь к файлам
    BASE = os.path.dirname(__file__)
    sub_path = os.path.join(BASE, 'submission.csv')
    docs_path = os.path.join(BASE, 'train_data.csv')
    eval_path = os.path.join(BASE, 'evaluation_metrics.csv')

    print('Loading submission and documents...')
    df_sub = pd.read_csv(sub_path, encoding='utf-8')
    df_docs = pd.read_csv(docs_path, encoding='utf-8')

    questions = df_sub['Вопрос'].fillna('').astype(str).tolist()
    preds = df_sub['Ответы на вопрос'].fillna('').astype(str).tolist()

    docs_texts = df_docs['text'].fillna('').astype(str).tolist()
    docs_ids = df_docs['id'].astype(str).tolist()

    # Находим наиболее релевантный документ к вопросу
    print('Computing TF-IDF and finding most relevant document for each question...')
    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
    doc_mat = tfidf.fit_transform(docs_texts)

    best_ids = []
    best_texts = []
    best_sims = []

    batch_size = 256
    n = len(questions)
    for start in tqdm(range(0, n, batch_size), desc='Find best doc batches'):
        end = min(n, start + batch_size)
        q_batch = questions[start:end]
        q_mat = tfidf.transform(q_batch)
        sims = cosine_similarity(q_mat, doc_mat)
        idx = sims.argmax(axis=1)
        vals = sims.max(axis=1)
        for i, j in enumerate(idx):
            best_ids.append(docs_ids[j])
            best_texts.append(docs_texts[j])
            best_sims.append(float(vals[i]))

    # Compute BERTScore in batches
    print('Computing BERTScore between predictions and best-doc texts...')
    bert_f1s = []
    batch_size = 64
    device = None  # bert-score will choose CPU if no CUDA
    for start in tqdm(range(0, n, batch_size), desc='BERTScore batches'):
        end = min(n, start + batch_size)
        preds_batch = preds[start:end]
        refs_batch = best_texts[start:end]
        P, R, F1 = bertscore_score(preds_batch, refs_batch, lang='ru', rescale_with_baseline=True)
        # F1 is a tensor; convert to floats
        if hasattr(F1, 'cpu'):
            f1_vals = F1.cpu().numpy().tolist()
        else:
            f1_vals = np.array(F1).tolist()
        bert_f1s.extend([float(x) for x in f1_vals])

    # Save results
    print('Saving results...')
    results = pd.DataFrame({
        'ID вопроса': df_sub.iloc[:,0],
        'Вопрос': questions,
        'best_doc_id': best_ids,
        'best_doc_similarity': best_sims,
        'bertscore_f1': bert_f1s,
        'Ответы на вопрос': preds
    })
    results.to_csv(os.path.join(BASE, 'bertscore_by_question.csv'), index=False, encoding='utf-8')

    # Try to merge with existing evaluation_metrics.csv (if lengths match)
    if os.path.exists(eval_path):
        df_eval = pd.read_csv(eval_path, encoding='utf-8')
        if len(df_eval) == len(results):
            df_eval['bertscore_f1'] = results['bertscore_f1']
            df_eval.to_csv(os.path.join(BASE, 'evaluation_metrics_with_bertscore.csv'), index=False, encoding='utf-8')
            print('Saved evaluation_metrics_with_bertscore.csv')
        else:
            print('evaluation_metrics.csv length does not match submission — saved separate bertscore_by_question.csv')
    else:
        print('evaluation_metrics.csv not found — saved bertscore_by_question.csv')

    print('Summary: average BERTScore F1 =', np.mean(bert_f1s))
