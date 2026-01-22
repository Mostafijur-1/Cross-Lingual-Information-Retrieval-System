import os
import json
import math
import re
from typing import List, Dict, Optional, Tuple
import time

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

try:
    import jellyfish
except Exception:
    jellyfish = None

try:
    from indic_transliteration.sanscript import transliterate, DEVANAGARI, ITRANS, HK, ISO
    from indic_transliteration import sanscript
except Exception:
    transliterate = None

from src.nlp_features import get_embedding


def simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if t]


def _char_ngrams(s: str, n: int = 3):
    s = re.sub(r"\s+", " ", s)
    s = f" {s} "
    return [s[i:i+n] for i in range(len(s)-n+1)]


def char_ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    A = set(_char_ngrams(a, n))
    B = set(_char_ngrams(b, n))
    if not A or not B:
        return 0.0
    inter = A.intersection(B)
    union = A.union(B)
    return float(len(inter)) / float(len(union))


class RetrievalEngine:
    def __init__(self, docs: List[Dict], index_dir: str = 'index'):
        self.docs = docs
        self.ids = list(range(len(docs)))
        self.texts = [((d.get('title','') or '') + ' ' + (d.get('body','') or '')).strip() for d in docs]
        self.langs = [d.get('language') for d in docs]

        # Try to load persisted TF-IDF and embeddings if available
        self.index_dir = index_dir
        tfidf_path = os.path.join(index_dir, 'tfidf_vectorizer.joblib')
        tfidf_matrix_path = os.path.join(index_dir, 'tfidf_matrix.npz')
        embeddings_path = os.path.join(index_dir, 'embeddings.npy')
        emb_mask_path = os.path.join(index_dir, 'emb_mask.npy')

        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        try:
            import joblib
            from scipy import sparse
            if os.path.exists(tfidf_path) and os.path.exists(tfidf_matrix_path):
                self.tfidf_vectorizer = joblib.load(tfidf_path)
                self.tfidf_matrix = sparse.load_npz(tfidf_matrix_path)
        except Exception:
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None

        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            # TF-IDF (build and keep in memory)
            self.tfidf_vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.texts)

        # BM25
        try:
            if BM25Okapi is not None:
                tokenized = [simple_tokenize(t) for t in self.texts]
                self.bm25 = BM25Okapi(tokenized)
            else:
                self.bm25 = None
        except Exception:
            self.bm25 = None

        # embeddings: try load persisted
        self.embeddings = []
        try:
            if os.path.exists(embeddings_path) and os.path.exists(emb_mask_path):
                emb_arr = np.load(embeddings_path)
                emb_mask = np.load(emb_mask_path)
                for i in range(len(emb_arr)):
                    if not emb_mask[i]:
                        self.embeddings.append(None)
                    else:
                        self.embeddings.append(np.array(emb_arr[i], dtype=float))
                # done
            else:
                raise FileNotFoundError()
        except Exception:
            # compute embeddings on-the-fly
            for t in self.texts:
                vec = get_embedding(t)
                if vec is None or len(vec) == 0:
                    self.embeddings.append(None)
                else:
                    self.embeddings.append(np.array(vec, dtype=float))

    def search_tfidf(self, query: str, topk: int = 10):
        qv = self.tfidf_vectorizer.transform([query])
        sims = cosine_similarity(qv, self.tfidf_matrix)[0]
        idx = np.argsort(-sims)[:topk]
        return [(int(i), float(sims[i])) for i in idx]

    def search_bm25(self, query: str, topk: int = 10):
        if self.bm25 is None:
            # fallback to TF-IDF
            return self.search_tfidf(query, topk)
        qtok = simple_tokenize(query)
        scores = self.bm25.get_scores(qtok)
        idx = np.argsort(-np.array(scores))[:topk]
        return [(int(i), float(scores[i])) for i in idx]

    def _fuzzy_score(self, a: str, b: str) -> float:
        if jellyfish is not None:
            try:
                return jellyfish.jaro_winkler_similarity(a, b)
            except Exception:
                pass
        # fallback to ratio using SequenceMatcher
        from difflib import SequenceMatcher
        return SequenceMatcher(None, a, b).ratio()

    def transliteration_score(self, query: str, text: str) -> float:
        # improved transliteration matching: compare Latin/ITRANS forms and char n-grams
        scores = []
        scores.append(self._fuzzy_score(query, text))
        # compute char n-gram Jaccard on raw text
        try:
            scores.append(char_ngram_jaccard(query.lower(), text.lower(), n=3))
        except Exception:
            pass

        if transliterate is not None:
            try:
                # normalize both sides to a Latin-compatible phonetic form (ITRANS) when possible
                def _to_latin(s: str) -> str:
                    # if contains non-ascii, try to transliterate from Bengali to ITRANS
                    if any(ord(ch) > 127 for ch in s):
                        try:
                            return transliterate(s, sanscript.BENGALI, sanscript.ITRANS)
                        except Exception:
                            return s
                    return s

                q_lat = _to_latin(query)
                t_lat = _to_latin(text)

                # fuzzy on transliterated Latin forms
                scores.append(self._fuzzy_score(q_lat.lower(), t_lat.lower()))
                # char n-gram Jaccard on transliterated forms -- helps cross-script matching
                scores.append(char_ngram_jaccard(q_lat.lower(), t_lat.lower(), n=3))
            except Exception:
                pass

        return max(scores)

    def search_fuzzy(self, query: str, topk: int = 10, use_translit: bool = True):
        scores = []
        for t in self.texts:
            a = query.lower()
            b = t.lower()
            sc = self._fuzzy_score(a, b)
            if use_translit:
                try:
                    sc = max(sc, self.transliteration_score(a, b))
                except Exception:
                    pass
            scores.append(sc)
        idx = np.argsort(-np.array(scores))[:topk]
        return [(int(i), float(scores[i])) for i in idx]

    def search_semantic(self, query: str, topk: int = 10):
        t0 = time.time()
        qvec = get_embedding(query)
        embed_time = (time.time() - t0)
        if qvec is None or len(qvec) == 0:
            return [], embed_time
        qv = np.array(qvec, dtype=float)
        sims = []
        for e in self.embeddings:
            if e is None:
                sims.append(-1.0)
            else:
                # cosine
                dot = np.dot(qv, e)
                denom = (np.linalg.norm(qv) * np.linalg.norm(e))
                sims.append((dot / denom) if denom != 0 else 0.0)
        idx = np.argsort(-np.array(sims))[:topk]
        return [(int(i), float(sims[i])) for i in idx], embed_time

    def _normalize_scores(self, scores: List[Tuple[int, float]]) -> Dict[int, float]:
        out: Dict[int, float] = {}
        if not scores:
            return out
        vals = np.array([s for (_, s) in scores], dtype=float)
        maxv = vals.max()
        if maxv <= 0:
            minv = vals.min()
            denom = (maxv - minv) if (maxv - minv) != 0 else 1.0
            for (i, s) in scores:
                out[i] = float((s - minv) / denom)
        else:
            for (i, s) in scores:
                out[i] = float(s / maxv)
        return out

    def rank(self, query: str, topk: int = 10, weights: Optional[Dict] = None) -> Tuple[List[Dict], Dict]:
        if weights is None:
            weights = {'bm25': 0.3, 'semantic': 0.5, 'fuzzy': 0.2, 'tfidf': 0.0}

        t_start = time.time()

        t0 = time.time()
        bm_scores = self.search_bm25(query, topk=len(self.texts))
        t_bm = time.time() - t0

        t0 = time.time()
        tf_scores = self.search_tfidf(query, topk=len(self.texts))
        t_tfidf = time.time() - t0

        t0 = time.time()
        fz_scores = self.search_fuzzy(query, topk=len(self.texts))
        t_fuzzy = time.time() - t0

        t0 = time.time()
        sem_res = self.search_semantic(query, topk=len(self.texts))
        if isinstance(sem_res, tuple):
            sem_scores, embed_time = sem_res
        else:
            sem_scores = sem_res
            embed_time = time.time() - t0
        t_sem = time.time() - t0

        norm_bm = self._normalize_scores(bm_scores)
        norm_tf = self._normalize_scores(tf_scores)
        norm_fz = self._normalize_scores(fz_scores)
        norm_sem = self._normalize_scores(sem_scores)

        candidates = set()
        for d in (list(norm_bm.keys()) + list(norm_tf.keys()) + list(norm_fz.keys()) + list(norm_sem.keys())):
            candidates.add(d)

        scores: Dict[int, Dict] = {}
        for doc_id in candidates:
            s_bm = norm_bm.get(doc_id, 0.0)
            s_tf = norm_tf.get(doc_id, 0.0)
            s_fz = norm_fz.get(doc_id, 0.0)
            s_sem = norm_sem.get(doc_id, 0.0)
            combined = weights.get('bm25', 0) * s_bm + weights.get('tfidf', 0) * s_tf + weights.get('fuzzy', 0) * s_fz + weights.get('semantic', 0) * s_sem
            scores[doc_id] = {
                'bm25': s_bm,
                'tfidf': s_tf,
                'fuzzy': s_fz,
                'semantic': s_sem,
                'score': combined
            }

        ranked = sorted(scores.items(), key=lambda kv: -kv[1]['score'])[:topk]

        total_time = (time.time() - t_start)

        results: List[Dict] = []
        for doc_id, md in ranked:
            results.append({'doc_id': int(doc_id), 'score': float(md['score']), 'bm25': md['bm25'], 'tfidf': md['tfidf'], 'fuzzy': md['fuzzy'], 'semantic': md['semantic']})

        timing = {
            'total_ms': total_time * 1000.0,
            'bm25_ms': t_bm * 1000.0,
            'tfidf_ms': t_tfidf * 1000.0,
            'fuzzy_ms': t_fuzzy * 1000.0,
            'semantic_ms': t_sem * 1000.0,
            'embed_ms': embed_time * 1000.0
        }

        return results, timing

    def search_hybrid(self, query: str, topk: int = 10, weights: Optional[Dict] = None):
        # weights: {'bm25':0.3,'semantic':0.5,'fuzzy':0.2}
        if weights is None:
            weights = {'bm25': 0.3, 'semantic': 0.5, 'fuzzy': 0.2}

        N = len(self.texts)
        score_matrix = np.zeros((N,))

        # BM25
        bm = np.zeros((N,))
        try:
            for i, s in self.search_bm25(query, topk=N):
                bm[i] = s
        except Exception:
            pass
        if bm.max() > 0:
            bm = bm / bm.max()

        # semantic
        sem = np.zeros((N,))
        try:
            for i, s in self.search_semantic(query, topk=N):
                sem[i] = s
        except Exception:
            pass
        if sem.max() > 0:
            sem = sem / sem.max()

        # fuzzy
        fz = np.zeros((N,))
        try:
            for i, s in self.search_fuzzy(query, topk=N):
                fz[i] = s
        except Exception:
            pass
        if fz.max() > 0:
            fz = fz / fz.max()

        score_matrix = weights.get('bm25', 0) * bm + weights.get('semantic', 0) * sem + weights.get('fuzzy', 0) * fz
        idx = np.argsort(-score_matrix)[:topk]
        return [(int(i), float(score_matrix[i])) for i in idx]


def load_dataset(path: str) -> List[Dict]:
    docs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                docs.append(json.loads(line))
            except Exception:
                continue
    return docs


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--query', required=True)
    args = ap.parse_args()
    docs = load_dataset(args.dataset)
    eng = RetrievalEngine(docs)
    print('TF-IDF top 5:')
    for i, s in eng.search_tfidf(args.query, 5):
        print(s, docs[i].get('title') or docs[i].get('url'))
    print('\nBM25 top 5:')
    for i, s in eng.search_bm25(args.query, 5):
        print(s, docs[i].get('title') or docs[i].get('url'))
    print('\nSemantic top 5:')
    for i, s in eng.search_semantic(args.query, 5):
        print(s, docs[i].get('title') or docs[i].get('url'))