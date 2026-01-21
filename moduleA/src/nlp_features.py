import spacy
import stanza
from sentence_transformers import SentenceTransformer

# Load models once with graceful fallbacks
try:
    nlp_en = spacy.load("en_core_web_sm")
except Exception:
    try:
        from spacy.cli import download as spacy_download
        spacy_download("en_core_web_sm")
        nlp_en = spacy.load("en_core_web_sm")
    except Exception:
        nlp_en = spacy.blank("en")

try:
    stanza.download("bn")
    nlp_bn = stanza.Pipeline("bn")
except Exception:
    nlp_bn = None

try:
    embedder = SentenceTransformer("distiluse-base-multilingual-cased")
except Exception:
    embedder = None

def extract_entities(text, lang):
    entities = []

    if lang == "en":
        doc = nlp_en(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]

    elif lang == "bn" and nlp_bn is not None:
        doc = nlp_bn(text)
        for sent in doc.sentences:
            for ent in sent.ents:
                entities.append((ent.text, ent.type))

    return entities

def get_embedding(text):
    if embedder is None:
        return []
    return embedder.encode(text).tolist()
