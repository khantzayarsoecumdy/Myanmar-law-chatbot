from pathlib import Path
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
    for path in files:
        content = read_file(path)
        chunks = split_text(content)
        if not chunks:
            print(f"WARNING: no text extracted from {path.name}; scanned PDFs may need OCR.")
            continue
        for idx, chunk in enumerate(chunks):
            ids.append(f"{path.name}-{idx}")
            texts.append(chunk)
            metas.append({"source": path.name, "chunk": idx + 1})
        print(f"Read {path.name}: {len(chunks)} chunks")

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

    # Build all new vectors before replacing the old collection. If Ollama is
    # unavailable or the model is missing, the existing index remains usable.
    try:
        client.delete_collection("law_documents")
    except Exception:
        pass
    collection = client.create_collection(
        name="law_documents",
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": len(embeddings[0]),
        },
    )
    for start in range(0, len(texts), batch_size):
        collection.add(
            ids=ids[start:start + batch_size],
            documents=texts[start:start + batch_size],
            metadatas=metas[start:start + batch_size],
            embeddings=embeddings[start:start + batch_size],
        )
    print(
        f"Indexed {len(texts)} chunks from {len(files)} document(s) with "
        f"{EMBEDDING_MODEL} ({len(embeddings[0])} dimensions) into {CHROMA_DIR}"
    )

if __name__ == "__main__":
    main()
