_model = None


def _get_model():
    global _model
    if _model is None:
        # Deferred: importing sentence_transformers pulls in torch, which is
        # the single largest contributor to this process's memory footprint.
        # Loading it at request time (only when an intake actually needs an
        # embedding) instead of at module import keeps app startup lean --
        # necessary to fit inside a memory-constrained deploy target.
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_embedding(text: str) -> list[float]:
    return _get_model().encode(text, normalize_embeddings=True).tolist()
