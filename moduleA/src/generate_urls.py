import sys
import time
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup


def normalize_url(base, link):
    return urljoin(base, link.split('#')[0])


def same_domain(a, b):
    return urlparse(a).netloc == urlparse(b).netloc


def is_valid_scheme(url):
    return urlparse(url).scheme in ("http", "https")


def looks_like_article(soup):
    paragraphs = soup.find_all('p')
    text = ' '.join(p.get_text() for p in paragraphs)
    return len(paragraphs) >= 3 and len(text) > 200


def generate(seed_file, out_file, max_urls=2500, max_pages=20000, delay=0.1):
    with open(seed_file, 'r', encoding='utf-8') as f:
        seeds = [l.strip() for l in f if l.strip()]

    visited = set()
    articles = []
    queue = list(seeds)

    while queue and len(articles) < max_urls and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, 'html.parser')

            # If page looks like article, add
            if looks_like_article(soup):
                if url not in articles:
                    articles.append(url)
                    print(f"[ARTICLE] {len(articles)}/{max_urls} -> {url}")
                    if len(articles) >= max_urls:
                        break

            # enqueue internal links
            for a in soup.find_all('a', href=True):
                link = normalize_url(url, a['href'])
                if not is_valid_scheme(link):
                    continue
                if same_domain(link, url) and link not in visited:
                    queue.append(link)

            time.sleep(delay)
        except Exception as e:
            print(f"[ERROR] {url} -> {e}")

    with open(out_file, 'w', encoding='utf-8') as out:
        for u in articles:
            out.write(u + '\n')

    print(f"Saved {len(articles)} URLs to {out_file}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python generate_urls.py <seed_file> <out_file> [max_urls]")
        sys.exit(1)
    seed = sys.argv[1]
    out = sys.argv[2]
    maxu = int(sys.argv[3]) if len(sys.argv) > 3 else 2500
    generate(seed, out, max_urls=maxu)
