

import re
import sys
import traceback


_KEEP_LABELS = {
    "PERSON": "PERSON",
    "ORG": "ORGANIZATION",
    "GPE": "GPE",
    "LOC": "LOCATION",
    "NORP": "GROUP",     
    "FAC": "FACILITY",
    "EVENT": "EVENT",
}

_nlp = None
_spacy_load_failed = False


def _get_spacy_model():
    """Lazily load the spaCy model once and cache it."""
    global _nlp, _spacy_load_failed

    if _nlp is not None or _spacy_load_failed:
        return _nlp

    try:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        _spacy_load_failed = True
        print(
            "[entity_extraction] spaCy model not available "
            f"({e}). Falling back to the NLTK chunker -- "
            "run `pip install spacy && python -m spacy download "
            "en_core_web_sm` for much better results.",
            file=sys.stderr
        )
        _nlp = None

    return _nlp


def _is_junk(entity_text):
    """Filter out one/two letter fragments and pure punctuation/numbers."""
    cleaned = entity_text.strip()

    if len(cleaned) < 2:
        return True

    if not re.search(r"[A-Za-z]", cleaned):
        return True

    return False


def _extract_with_spacy(text, nlp):
    entities = []

    doc = nlp(text)

    for ent in doc.ents:

        label = _KEEP_LABELS.get(ent.label_)

        if not label:
            continue

        entity_text = ent.text.strip()

        if _is_junk(entity_text):
            continue

        entities.append((entity_text, label))

    return entities


def _extract_with_nltk(text):
    """Legacy fallback, kept only for when spaCy isn't installed."""
    import nltk

    entities = []

    try:
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        chunks = nltk.ne_chunk(pos_tags)

        for chunk in chunks:

            if hasattr(chunk, "label"):

                entity = " ".join(c[0] for c in chunk)

                if _is_junk(entity):
                    continue

                entities.append((entity, chunk.label()))

    except LookupError as e:

        print(
            "[entity_extraction] Missing NLTK resource - "
            f"run nltk.download() for it. Details: {e}",
            file=sys.stderr
        )

    except Exception:
        print(
            "[entity_extraction] Unexpected error during NER:",
            file=sys.stderr
        )
        traceback.print_exc()

    return entities


def extract_entities(text):

    if not text or not text.strip():
        return []

    nlp = _get_spacy_model()

    try:
        if nlp is not None:
            entities = _extract_with_spacy(text, nlp)
        else:
            entities = _extract_with_nltk(text)

    except Exception:
        print(
            "[entity_extraction] Unexpected error during NER:",
            file=sys.stderr
        )
        traceback.print_exc()
        return []

   
    seen = {}

    for entity_text, entity_type in entities:
        key = (entity_text.lower(), entity_type)
        if key not in seen:
            seen[key] = (entity_text, entity_type)

    return list(seen.values())
