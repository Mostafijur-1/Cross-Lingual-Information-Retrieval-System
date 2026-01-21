from src.build_dataset import process_urls
from src.build_index import build_inverted_index, save_index

DATASET = "data/processed/dataset.jsonl"

# Step 1: Build dataset (use full URL lists)
process_urls("data/raw/bn_urls_full.txt", DATASET)
process_urls("data/raw/en_urls_full.txt", DATASET)

# Step 2: Build index
index = build_inverted_index(DATASET)
save_index(index, "index/inverted_index.pkl")

print("✅ Pipeline completed successfully")
