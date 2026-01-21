import json
from tqdm import tqdm
from src.crawler import crawl_article
from src.preprocess import build_metadata
from src.nlp_features import extract_entities, get_embedding

def process_urls(url_file, output_file):
    with open(url_file, "r", encoding="utf-8") as f:
        urls = [u.strip() for u in f if u.strip()]

    with open(output_file, "a", encoding="utf-8") as out:
        for url in tqdm(urls):
            title, body, method = crawl_article(url)
            if not body:
                continue

            doc = build_metadata(title, body, url)
            doc["crawler"] = method

            doc["named_entities"] = extract_entities(doc["body"], doc["language"])
            doc["embedding"] = get_embedding(doc["body"])

            json.dump(doc, out, ensure_ascii=False)
            out.write("\n")
