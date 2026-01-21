import pickle
import json

def load_index(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def search(query, index, dataset_file):
    results = set()

    for word in query.lower().split():
        if word in index:
            results.update(index[word])

    with open(dataset_file, "r", encoding="utf-8") as f:
        docs = list(f)

    return [json.loads(docs[i]) for i in results]
