import os
import sys
from flask import Flask, request, render_template, redirect, url_for

# ensure module root is importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.retrieval import load_dataset, RetrievalEngine

DATASET_PATH = os.path.join(ROOT, 'data', 'processed', 'dataset.jsonl')

app = Flask(__name__, template_folder='templates')

print('Loading dataset and retrieval engine...')
docs = load_dataset(DATASET_PATH)
# for responsiveness, limit size if very large
if len(docs) > 5000:
    docs = docs[:5000]
engine = RetrievalEngine(docs)
print('Ready — docs:', len(docs))


@app.route('/')
def index():
    return render_template('search.html')


@app.route('/search', methods=['GET','POST'])
def search():
    q = request.values.get('q','').strip()
    if not q:
        return redirect(url_for('index'))

    topk = int(request.values.get('topk', 5))
    # use rank() to get normalized combined scores and timings
    ranked, timing = engine.rank(q, topk=topk)

    docs_map = engine.docs

    rendered = []
    for r in ranked:
        d = docs_map[r['doc_id']]
        rendered.append({
            'id': r['doc_id'],
            'score': r['score'],
            'title': d.get('title') or d.get('url'),
            'url': d.get('url'),
            'snippet': (d.get('body') or '')[:400],
            'bm25': r.get('bm25', 0.0),
            'tfidf': r.get('tfidf', 0.0),
            'fuzzy': r.get('fuzzy', 0.0),
            'semantic': r.get('semantic', 0.0)
        })

    # low-confidence warning
    low_confidence = False
    if len(rendered) > 0 and rendered[0]['score'] < 0.20:
        low_confidence = True

    return render_template('results.html', query=q, results=rendered, timing=timing, low_confidence=low_confidence)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)
