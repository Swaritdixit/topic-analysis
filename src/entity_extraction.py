import nltk

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

    except Exception:
        pass

    return list(set(entities))