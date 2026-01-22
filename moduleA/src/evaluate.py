import csv
import json
import math
from typing import Dict, List, Set

from src.retrieval import load_dataset, RetrievalEngine


def precision_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    if k == 0:
        return 0.0
    return sum(1 for d in retrieved[:k] if d in relevant) / float(k)


def recall_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for d in retrieved[:k] if d in relevant) / float(len(relevant))


def dcg_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    dcg = 0.0
    for i, d in enumerate(retrieved[:k]):
        rel = 1.0 if d in relevant else 0.0
        denom = math.log2(i+2)
        dcg += rel / denom
    return dcg


def idcg_at_k(relevant: Set[int], k: int) -> float:
    # ideal DCG: all relevant docs at top
    rels = min(len(relevant), k)
    idcg = 0.0
    for i in range(rels):
        idcg += 1.0 / math.log2(i+2)
    return idcg


def ndcg_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    idcg = idcg_at_k(relevant, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(retrieved, relevant, k) / idcg


def mrr(retrieved: List[int], relevant: Set[int]) -> float:
    for i, d in enumerate(retrieved):
        if d in relevant:
            return 1.0 / (i+1)
    return 0.0


def load_qrels(path: str):
    # expects CSV with columns: query,doc_url,language,yesno,annotator
    qrels = {}
    with open(path, 'r', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            q = row['query'].strip()
            url = row['doc_url'].strip()
            yes = row.get('yesno','').strip().lower() in ('yes','1','true','y')
            if q not in qrels:
                qrels[q] = {}
            qrels[q][url] = yes
    return qrels


def evaluate(dataset_path: str, qrels_csv: str):
    docs = load_dataset(dataset_path)
    url_to_id = {d.get('url'): idx for idx, d in enumerate(docs)}
    engine = RetrievalEngine(docs)

    qrels = load_qrels(qrels_csv)

    results = []
    for q, mapping in qrels.items():
        relevant_urls = {u for u, v in mapping.items() if v}
        relevant_ids = {url_to_id[u] for u in relevant_urls if u in url_to_id}
        ranked, timing = engine.rank(q, topk=100)
        retrieved = [r['doc_id'] for r in ranked]

        p10 = precision_at_k(retrieved, relevant_ids, 10)
        r50 = recall_at_k(retrieved, relevant_ids, 50)
        ndcg10 = ndcg_at_k(retrieved, relevant_ids, 10)
        q_mrr = mrr(retrieved, relevant_ids)

        results.append({
            'query': q,
            'p10': p10,
            'r50': r50,
            'ndcg10': ndcg10,
            'mrr': q_mrr,
            'timing_ms': timing['total_ms']
        })

    # aggregate
    agg = {'p10': 0.0, 'r50': 0.0, 'ndcg10': 0.0, 'mrr': 0.0}
    for r in results:
        agg['p10'] += r['p10']
        agg['r50'] += r['r50']
        agg['ndcg10'] += r['ndcg10']
        agg['mrr'] += r['mrr']
    n = len(results) if results else 1
    agg = {k: v / n for k, v in agg.items()}

    print('Queries evaluated:', len(results))
    print('Avg Precision@10:', agg['p10'])
    print('Avg Recall@50:', agg['r50'])
    print('Avg nDCG@10:', agg['ndcg10'])
    print('Avg MRR:', agg['mrr'])

    return results, agg


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='data/processed/dataset.jsonl')
    ap.add_argument('--qrels', required=True)
    args = ap.parse_args()
    evaluate(args.dataset, args.qrels)
