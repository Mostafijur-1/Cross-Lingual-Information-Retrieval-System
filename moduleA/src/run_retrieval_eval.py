import os
import json
from pprint import pprint

from src.retrieval import load_dataset, RetrievalEngine


def sample_queries():
    return [
        # English
        'Bangladesh cricket victory',
        'election results in Bangladesh',
        # Bangla (UTF-8)
        'বাংলাদেশ ক্রিকেট জয়',
        'বাংলাদেশে নির্বাচন ফলাফল',
        # transliterated Latin representing Bangla terms
        'Bangladesh cricketeer',
        'Bangladesh nirbachon'
    ]


def run_eval(dataset_path: str, topk: int = 5):
    print('Loading dataset...', dataset_path)
    docs = load_dataset(dataset_path)
    print('Docs loaded:', len(docs))
    # restrict for speed if huge
    if len(docs) > 5000:
        docs = docs[:5000]
        print('Using first 5000 docs for evaluation (speed).')

    engine = RetrievalEngine(docs)

    for q in sample_queries():
        print('\n' + '='*80)
        print('Query:', q)
        print('\n-- TF-IDF --')
        for i, s in engine.search_tfidf(q, topk):
            print(f'{s:.4f}', docs[i].get('title') or docs[i].get('url'))

        print('\n-- BM25 --')
        for i, s in engine.search_bm25(q, topk):
            print(f'{s:.4f}', docs[i].get('title') or docs[i].get('url'))

        print('\n-- Fuzzy/Transliteration --')
        for i, s in engine.search_fuzzy(q, topk):
            print(f'{s:.4f}', docs[i].get('title') or docs[i].get('url'))

        print('\n-- Semantic (embeddings) --')
        sem = engine.search_semantic(q, topk)
        if not sem:
            print('No embeddings available or embedding failed for query.')
        for i, s in sem:
            print(f'{s:.4f}', docs[i].get('title') or docs[i].get('url'))

        print('\n-- Hybrid (0.3 BM25 + 0.5 SEM + 0.2 FUZZY) --')
        for i, s in engine.search_hybrid(q, topk, weights={'bm25':0.3,'semantic':0.5,'fuzzy':0.2}):
            print(f'{s:.4f}', docs[i].get('title') or docs[i].get('url'))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=False, default='data/processed/dataset.jsonl')
    ap.add_argument('--topk', type=int, default=5)
    args = ap.parse_args()
    run_eval(args.dataset, args.topk)
