import json
import pickle
from collections import defaultdict

def build_inverted_index(dataset_file):
    index = defaultdict(list)

    with open(dataset_file, "r", encoding="utf-8") as f:
        for doc_id, line in enumerate(f):
            doc = json.loads(line)
            for word in doc["body"].lower().split():
                index[word].append(doc_id)

    return index

def save_index(index, path):
    with open(path, "wb") as f:
        pickle.dump(index, f)
