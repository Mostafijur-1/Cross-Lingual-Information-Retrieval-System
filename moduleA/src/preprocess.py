from langdetect import detect
from datetime import datetime

def clean_text(text):
    return " ".join(text.split())

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def token_count(text):
    return len(text.split())

def build_metadata(title, body, url):
    body = clean_text(body)
    return {
        "title": title,
        "body": body,
        "url": url,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "language": detect_language(body),
        "tokens": token_count(body)
    }
