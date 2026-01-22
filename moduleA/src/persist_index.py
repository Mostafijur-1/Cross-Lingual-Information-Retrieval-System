import os
import json
import numpy as np
from typing import List, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from scipy import sparse

from src.nlp_features import get_embedding


def load_dataset(path: str) -> List[Dict]:
    docs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                docs.append(json.loads(line))
            except Exception:
                continue
    return docs


def build_and_persist(dataset_path: str, out_dir: str = 'index'):
    os.makedirs(out_dir, exist_ok=True)
    docs = load_dataset(dataset_path)
    texts = [((d.get('title','') or '') + ' ' + (d.get('body','') or '')).strip() for d in docs]

    print('Fitting TF-IDF...')
    tf = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    tf_matrix = tf.fit_transform(texts)
    joblib.dump(tf, os.path.join(out_dir, 'tfidf_vectorizer.joblib'))
    sparse.save_npz(os.path.join(out_dir, 'tfidf_matrix.npz'), tf_matrix)
    print('TF-IDF saved.')

    print('Computing embeddings (this may take a while)...')
    emb_list = []
    mask = []
    for i, t in enumerate(texts):
        v = get_embedding(t)
        if v is None or len(v) == 0:
            emb_list.append(np.zeros(1, dtype=float))
            mask.append(False)
        else:
            emb_list.append(np.array(v, dtype=float))
            mask.append(True)
        if (i+1) % 100 == 0:
            print('Computed embeddings for', i+1, 'docs')

    # pad variable-length embeddings into 2D array if necessary
    dims = max((e.shape[0] for e in emb_list if e.size>1), default=0)
    if dims == 0:
        emb_arr = np.zeros((len(emb_list), 1), dtype=float)
    else:
        emb_arr = np.zeros((len(emb_list), dims), dtype=float)
        for i, e in enumerate(emb_list):
            if mask[i] and e.size == dims:
                emb_arr[i] = e
            elif mask[i] and e.size < dims:
                emb_arr[i, :e.size] = e

    np.save(os.path.join(out_dir, 'embeddings.npy'), emb_arr)
    np.save(os.path.join(out_dir, 'emb_mask.npy'), np.array(mask, dtype=bool))
    print('Embeddings persisted to', out_dir)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='data/processed/dataset.jsonl')
    ap.add_argument('--out', default='index')
    args = ap.parse_args()
    build_and_persist(args.dataset, args.out)
