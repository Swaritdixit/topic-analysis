import nltk
import sys
import traceback


def extract_entities(text):

    entities = []

    try:
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        chunks = nltk.ne_chunk(pos_tags)

        for chunk in chunks:

            if hasattr(chunk, "label"):

                entity = " ".join(
                    c[0]
                    for c in chunk
                )

                entity_type = chunk.label()

                entities.append(
                    (
                        entity,
                        entity_type
                    )
                )

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


    return list(dict.fromkeys(entities))
