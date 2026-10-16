from pathlib import Path
import hashlib
import chromadb
from app.config import DOCS_DIR, CHROMA_DIR, EMBEDDING_MODEL
from app.embedding import EmbeddingError, embed_texts
from app.documents import SUPPORTED, read_file, split_text

def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in DOCS_DIR.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    if not files:
        print(f"No PDF/DOCX/TXT files found in: {DOCS_DIR}")
        return

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    ids, texts, metas = [], [], []
    indexed_sources = set()
    for path in files:
        content = read_file(path)
        chunks = split_text(content)
        if not chunks:
            print(f"WARNING: no text extracted from {path.name}; scanned PDFs may need OCR.")
            continue
        source = path.relative_to(DOCS_DIR).as_posix()
        source_id = hashlib.sha256(source.encode("utf-8")).hexdigest()[:20]
        indexed_sources.add(source)
        for idx, chunk in enumerate(chunks):
            ids.append(f"{source_id}-{idx + 1}")
            texts.append(chunk)
            metas.append({"source": source, "chunk": idx + 1})
        print(f"Read {source}: {len(chunks)} chunks")

    if not texts:
        print("No readable text was found. Check your documents.")
        return

    batch_size = 64
    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start+batch_size]
        try:
            embeddings.extend(embed_texts(batch_texts))
        except EmbeddingError as exc:
            print(f"Embedding failed; existing index was left untouched: {exc}")
            return

    if len(embeddings) != len(texts):
        print("Embedding count did not match the document count; existing index was left untouched.")
        return

    expected_metadata = {
        "hnsw:space": "cosine",
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": len(embeddings[0]),
    }
    try:
        collection = client.get_collection("law_documents")
    except Exception:
        collection = None

    if collection is not None:
        current_metadata = collection.metadata or {}
        if any(current_metadata.get(key) != value for key, value in expected_metadata.items()):
            client.delete_collection("law_documents")
            collection = None

    if collection is None:
        collection = client.create_collection(
            name="law_documents",
            metadata=expected_metadata,
        )

    # Upsert in place so an API process that is already running can see new
    # documents without holding a stale collection object after a rebuild.
    for start in range(0, len(texts), batch_size):
        collection.upsert(
            ids=ids[start:start + batch_size],
            documents=texts[start:start + batch_size],
            metadatas=metas[start:start + batch_size],
            embeddings=embeddings[start:start + batch_size],
        )

    desired_ids = set(ids)
    existing = collection.get(include=["metadatas"])
    stale_ids = [record_id for record_id in existing["ids"] if record_id not in desired_ids]
    for start in range(0, len(stale_ids), batch_size):
        collection.delete(ids=stale_ids[start:start + batch_size])

    print(
        f"Indexed {len(texts)} chunks from {len(indexed_sources)} document(s) with "
        f"{EMBEDDING_MODEL} ({len(embeddings[0])} dimensions) into {CHROMA_DIR}"
    )

if __name__ == "__main__":
    main()
