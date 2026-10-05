"""
CRRA Lab C1 - Policy Knowledge Base Setup

Reads the BizOps procurement policy articles in data/kb/, splits each one into
one chunk per '## ' rule, and loads them into a persistent ChromaDB collection
called 'crra_policy' so the Analysis Agent (Lab C3) can cite policy.

Run from the project root:
    python data/kb_setup.py
"""

from pathlib import Path

import chromadb

DATA_DIR = Path(__file__).resolve().parent
KB_DIR = DATA_DIR / "kb"
CHROMA_DIR = DATA_DIR / "chroma_db"      # on-disk store, reused by later labs
COLLECTION_NAME = "crra_policy"

TEST_QUERIES = [
    "who approves a 60 lakh contract",
    "the contract auto-renews in two weeks and we missed the notice deadline",
    "two monitoring tools are both at 35 percent licence usage",
    "vendor is asking for an 18 percent price increase at renewal",
    "we no longer need this tool, what must we check before serving termination notice",
    # --- Deliberate break tests ---
    # Not covered by any policy: vector search still returns its nearest
    # neighbour, so watch for a LOW confidence score here.
    "what is the travel expense limit",
    # Spans two articles: approval bands (90 lakh = Band C) and termination
    # (early-exit penalties). Only one chunk wins with n_results=1.
    "can I terminate a 90 lakh contract early",
]


def chunk_markdown(text: str, filename: str) -> list[dict]:
    """Split one article into chunks at '## ' headings.

    The '# ' title line is skipped. Any text before the first '## ' heading
    is ignored, since every rule in this KB sits under its own heading.
    """
    chunks: list[dict] = []
    heading: str | None = None
    body: list[str] = []

    def flush() -> None:
        content = "\n".join(body).strip()
        if heading and content:
            chunks.append({"heading": heading, "body": content})

    for line in text.splitlines():
        if line.startswith("## "):
            flush()
            heading, body = line[3:].strip(), []
        elif line.startswith("# "):
            continue
        else:
            body.append(line)
    flush()

    stem = Path(filename).stem
    return [
        {
            "id": f"{stem}::{i:02d}",
            # Heading goes into the embedded text too: it is the most
            # descriptive line of the rule and improves retrieval.
            "document": f"{c['heading']}\n\n{c['body']}",
            "metadata": {"source": filename, "heading": c["heading"], "chunk_index": i},
        }
        for i, c in enumerate(chunks)
    ]


def build_collection(client):
    """Drop and recreate the collection so re-runs never duplicate chunks."""
    existing = [getattr(c, "name", c) for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    # Cosine space makes (1 - distance) a cosine similarity, so it reads
    # sensibly as a confidence score. The default (L2) does not.
    return client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def main() -> None:
    md_files = sorted(KB_DIR.glob("*.md"))
    if not md_files:
        raise SystemExit(f"No .md files found in {KB_DIR}")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = build_collection(client)

    ids, docs, metas = [], [], []
    print("Loading policy articles")
    print("=" * 60)
    for md in md_files:
        chunks = chunk_markdown(md.read_text(encoding="utf-8"), md.name)
        print(f"  {md.name:<34}{len(chunks):>3} chunks")
        for c in chunks:
            ids.append(c["id"])
            docs.append(c["document"])
            metas.append(c["metadata"])

    collection.add(ids=ids, documents=docs, metadatas=metas)
    print("-" * 60)
    print(f"  {'TOTAL':<34}{len(ids):>3} chunks from {len(md_files)} files")
    print(f"  Collection '{COLLECTION_NAME}' now holds {collection.count()} chunks")
    print(f"  Stored at {CHROMA_DIR}\n")

    print("Test queries (best match)")
    print("=" * 60)
    for q in TEST_QUERIES:
        res = collection.query(query_texts=[q], n_results=1)
        meta = res["metadatas"][0][0]
        confidence = 1 - res["distances"][0][0]
        print(f'  Q: "{q}"')
        print(f"     source:     {meta['source']}")
        print(f"     section:    {meta['heading']}")
        print(f"     confidence: {confidence:.2f}\n")


if __name__ == "__main__":
    main()