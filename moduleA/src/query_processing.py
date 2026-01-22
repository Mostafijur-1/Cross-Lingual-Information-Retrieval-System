import os
import sys
import re
from langdetect import detect, DetectorFactory
# ensure project root is on sys.path so `from src...` imports work when run as a script
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
DetectorFactory.seed = 0

try:
    from googletrans import Translator
    _translator = Translator()
except Exception:
    _translator = None

try:
    import nltk
    from nltk.corpus import wordnet as wn
    nltk_available = True
except Exception:
    nltk_available = False

try:
    from src.nlp_features import extract_entities
except Exception:
    def extract_entities(text, lang):
        return []


def detect_language(text):
    try:
        lang = detect(text)
        if lang.startswith('bn'):
            return 'bn'
        if lang.startswith('en'):
            return 'en'
        return lang
    except Exception:
        return 'en'


def normalize(text):
    if not isinstance(text, str):
        return ''
    s = text.strip()
    s = re.sub(r"\s+", ' ', s)
    try:
        s = s.lower()
    except Exception:
        pass
    return s


def translate_text(text, src='auto', tgt='en'):
    if not text:
        return ''
    if _translator is None:
        return text
    try:
        res = _translator.translate(text, src=src, dest=tgt)
        return res.text
    except Exception:
        return text


def expand_query(text, lang='en', max_synonyms=2):
    tokens = [t for t in re.findall(r"\w+", text)]
    expansions = set(tokens)
    if lang == 'en' and nltk_available:
        try:
            for t in tokens:
                syns = wn.synsets(t)
                for s in syns[:max_synonyms]:
                    for l in s.lemmas()[:max_synonyms]:
                        expansions.add(l.name().replace('_', ' '))
        except Exception:
            pass
    return list(expansions)


def map_named_entities(text, src_lang, tgt_lang):
    nes = extract_entities(text, src_lang)
    mapped = []
    for ent_text, ent_label in nes:
        translated = ent_text
        if src_lang != tgt_lang:
            translated = translate_text(ent_text, src=src_lang, tgt=tgt_lang)
        mapped.append({'text': ent_text, 'label': ent_label, 'mapped': translated})
    return mapped


def process_query(query, expand=True, map_ne=True, translate_targets=('en', 'bn')):
    """Process a query and return structured information useful for cross-lingual retrieval.

    Returns a dict with:
      - detected_language
      - normalized
      - translations: {lang: text}
      - expansions: {lang: [terms]}
      - named_entities: list of {text,label,mapped}
    """
    q = query or ''
    detected = detect_language(q)
    normalized = normalize(q)

    translations = {detected: normalized}
    for tgt in translate_targets:
        if tgt == detected:
            continue
        translations[tgt] = translate_text(normalized, src=detected, tgt=tgt)

    expansions = {}
    for lang, txt in translations.items():
        if expand:
            expansions[lang] = expand_query(txt, lang=lang)
        else:
            expansions[lang] = [txt]

    named_entities = []
    if map_ne:
        # map entities to all requested target langs (use translations keys)
        for tgt in translations.keys():
            mapped = map_named_entities(translations[tgt], detected, tgt)
            if mapped:
                named_entities.extend(mapped)

    return {
        'detected_language': detected,
        'normalized': normalized,
        'translations': translations,
        'expansions': expansions,
        'named_entities': named_entities,
    }


if __name__ == '__main__':
    import sys, json
    if len(sys.argv) < 2:
        print('Usage: python query_processing.py "your query here"')
        sys.exit(1)
    q = ' '.join(sys.argv[1:])
    out = process_query(q)
    print(json.dumps(out, ensure_ascii=False, indent=2))
