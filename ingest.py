"""Ingestion pipeline v2.

v2 changes vs v1:
  - split_and_tag(): adds source (clean filename) and chunk_id metadata to every
    chunk — required for [Source: filename] citation enforcement in the RAG prompt.
  - build_and_persist(): after saving to Chroma, pickles the raw Document list
    to bm25_index.pkl so the BM25 retriever can be rebuilt at startup without
    re-embedding.

Run: python ingest.py
  Step 1: load .txt / .pdf / .docx from docs/
  Step 2: split into chunks and tag with source + chunk_id
  Step 3: embed → Chroma (vector search) + pickle → bm25_index.pkl (keyword search)
"""

import os
import pickle
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.vectorstore import get_embeddings

try:
    from langchain_community.document_loaders.word_document import Docx2txtLoader
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


def load_documents(docs_dir: str) -> list[Document]:
    """Load all supported documents from the docs folder."""
    docs = []
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        print(f"  Docs folder '{docs_dir}' not found. Creating it.")
        docs_path.mkdir(parents=True)
        return docs

    for file_path in sorted(docs_path.iterdir()):
        if file_path.suffix == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs.extend(loader.load())
            print(f"  Loaded: {file_path.name}")
        elif file_path.suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            docs.extend(loader.load())
            print(f"  Loaded: {file_path.name}")
        elif file_path.suffix == ".docx" and DOCX_AVAILABLE:
            loader = Docx2txtLoader(str(file_path))
            docs.extend(loader.load())
            print(f"  Loaded: {file_path.name}")
        else:
            print(f"  Skipped: {file_path.name} (unsupported type)")

    return docs


def split_and_tag(docs: list[Document]) -> list[Document]:
    """Split documents into chunks and tag with source + chunk_id metadata.

    v2 change: the metadata is used in two places:
      1. _format_context() in graph.py labels each chunk [Source: filename, Chunk N]
      2. The RAG prompt instructs the LLM to cite [Source: filename] in its answer
      3. evaluate.py measures citation recall against these source names

    chunk_id resets to 0 for each source document so IDs are scoped per file.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    for doc in docs:
        chunks = splitter.split_documents([doc])
        source_name = Path(doc.metadata.get("source", "unknown")).name
        for i, chunk in enumerate(chunks):
            chunk.metadata["source"] = source_name
            chunk.metadata["chunk_id"] = i
        all_chunks.extend(chunks)

    return all_chunks


def build_and_persist(chunks: list[Document]) -> None:
    """Embed chunks → Chroma. Pickle chunks → bm25_index.pkl."""
    import shutil

    # Chroma
    if os.path.exists(settings.persist_dir):
        shutil.rmtree(settings.persist_dir)
        print(f"  Cleared existing Chroma at {settings.persist_dir}")

    embeddings = get_embeddings()
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="production_rag_v2",
        persist_directory=settings.persist_dir,
    )
    print(f"  Saved {len(chunks)} chunks to Chroma at {settings.persist_dir}")

    # BM25 — v2: pickle the raw Document list so BM25Retriever can be rebuilt
    with open(settings.bm25_index_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"  Saved BM25 index ({len(chunks)} docs) to {settings.bm25_index_path}")


def main():
    print("\nIngestion pipeline v2")
    print(f"  Docs dir  : {settings.docs_dir}")
    print(f"  Chroma    : {settings.persist_dir}")
    print(f"  BM25 index: {settings.bm25_index_path}")
    print(f"  Chunk size: {settings.chunk_size}, overlap: {settings.chunk_overlap}")
    print()

    print("Step 1: Loading documents...")
    docs = load_documents(settings.docs_dir)
    if not docs:
        print("  No documents found. Add .txt/.pdf/.docx files to ./docs and re-run.")
        return
    print(f"  Loaded {len(docs)} document(s)")

    print("\nStep 2: Splitting and tagging chunks...")
    chunks = split_and_tag(docs)
    print(f"  Created {len(chunks)} chunks with source + chunk_id metadata")

    print("\nStep 3: Embedding → Chroma + BM25 index...")
    print("  (First run downloads ONNX model ~40MB — subsequent runs are instant)")
    build_and_persist(chunks)

    print("\nIngestion complete.")
    print("Run `uvicorn main:app --reload` to start the API.")


if __name__ == "__main__":
    main()
