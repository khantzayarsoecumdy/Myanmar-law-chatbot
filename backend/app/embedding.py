import requests
from app.config import EMBEDDING_MODEL, OLLAMA_EMBED_URL


class EmbeddingError(Exception):
    pass


def embed_texts(texts: list[str], timeout: int = 300) -> list[list[float]]:
    if not texts:
        return []
    try:
        response = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": EMBEDDING_MODEL, "input": texts, "truncate": False},
            timeout=timeout,
        )
    except requests.Timeout as exc:
        raise EmbeddingError(f"Ollama embedding model {EMBEDDING_MODEL} timed out.") from exc
    except requests.ConnectionError as exc:
        raise EmbeddingError(
            f"Cannot connect to Ollama embeddings at {OLLAMA_EMBED_URL}. "
            f"Start Ollama and make sure {EMBEDDING_MODEL} is installed."
        ) from exc

    if not response.ok:
        raise EmbeddingError(
            f"Ollama embeddings returned HTTP {response.status_code}: {response.text[:500]}"
        )
    try:
        body = response.json()
    except ValueError as exc:
        raise EmbeddingError("Ollama embeddings returned a non-JSON response.") from exc

    embeddings = body.get("embeddings") if isinstance(body, dict) else None
    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
        raise EmbeddingError("Ollama embeddings returned the wrong number of vectors.")
    if not embeddings or not isinstance(embeddings[0], list) or not embeddings[0]:
        raise EmbeddingError("Ollama embeddings returned an empty vector.")
    dimension = len(embeddings[0])
    if any(not isinstance(vector, list) or len(vector) != dimension for vector in embeddings):
        raise EmbeddingError("Ollama embeddings returned vectors with inconsistent dimensions.")
    return embeddings
