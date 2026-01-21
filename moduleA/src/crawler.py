import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


def _is_valid_scheme(u):
    return urlparse(u).scheme in ("http", "https")


def _find_feed_urls(soup, base):
    feeds = []
    for link in soup.find_all('link', href=True):
        t = link.get('type', '')
        if 'rss' in t or 'xml' in t or 'atom' in t:
            feeds.append(urljoin(base, link['href']))
    # common feed paths
    parsed = urlparse(base)
    root = f"{parsed.scheme}://{parsed.netloc}"
    feeds.extend([urljoin(root, p) for p in ['/feed', '/rss', '/rss.xml', '/feed.xml']])
    return list(dict.fromkeys(feeds))


def _try_fetch_feed(feed_url):
    try:
        r = requests.get(feed_url, timeout=10)
        if r.status_code != 200:
            return None
        fsoup = BeautifulSoup(r.text, 'xml')
        item = fsoup.find('item') or fsoup.find('entry')
        if not item:
            return None
        title = item.find('title').get_text(strip=True) if item.find('title') else ''
        content = item.find('content:encoded') or item.find('description') or item.find('summary')
        if content:
            body = content.get_text()
        else:
            # fallback: join paragraph tags inside item
            paragraphs = item.find_all('p')
            body = ' '.join(p.get_text() for p in paragraphs)
        return title, body
    except Exception:
        return None


def crawl_article(url):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        paragraphs = soup.find_all("p")
        body = " ".join(p.get_text() for p in paragraphs)

        if body and len(body) > 200:
            return title, body, 'requests'

        # Try RSS/Atom feeds as fallback
        feeds = _find_feed_urls(soup, url)
        for f in feeds:
            if not _is_valid_scheme(f):
                continue
            res = _try_fetch_feed(f)
            if res:
                ft, fb = res
                return ft or title, fb or body, 'rss'

        return None, None, 'failed'
    except Exception as e:
        print(f"[ERROR] {url} -> {e}")
        return None, None, 'failed'
