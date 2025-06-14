from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from vendor_dsparse import parse_and_chunk_text
import openai
import hashlib
import streamlit as st

# ---- CONFIG ----
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = st.secrets["PINECONE_ENVIRONMENT"]
INDEX_NAME = "grant-rag"
DOC_DIR = "Grants"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

# ---- PINECONE INIT ----
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(INDEX_NAME, dimension=1536, metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT))
index = pc.Index(INDEX_NAME)
vectorstore = PineconeVectorStore(
    index=index,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
)

# ---- UTILS ----
def get_file_hash(fp: Path) -> str:
    sha = hashlib.sha256()
    with fp.open("rb") as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            sha.update(data)
    return sha.hexdigest()[:10]

def batchify(lst, size=100):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

# ---- MAIN ----
def main():
    all_chunks_with_ids = []

    for path in Path(DOC_DIR).rglob("*"):
        if path.suffix.lower() not in [".pdf", ".docx"]:
            continue

        # ✅ Skip if preview already exists
        preview_dir = Path("chunks_preview") / path.stem
        if preview_dir.exists():
            print(f"⏭️ Skipping already processed file: {path.name}")
            continue

        print(f"\n🔍 Processing: {path}")
        loader = PyPDFLoader(str(path)) if path.suffix.lower() == ".pdf" else UnstructuredWordDocumentLoader(str(path))

        try:
            documents = loader.load()
        except Exception as e:
            print(f"❌ Failed to load {path.name}: {e}")
            continue

        file_hash = get_file_hash(path)
        chunk_id = 0

        for doc in documents:
            text = doc.page_content or ""
            if not text.strip():
                print(f"⚠️ Skipped empty document inside {path.name}")
                continue

            chunks = parse_and_chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            if not chunks:
                print(f"❌ No chunks generated from {path.name}")
                continue

            print(f"➡️ First few chunks from {path.name}:")
            for i, chk in enumerate(chunks[:5]):
                preview = chk.page_content.replace("\n", " ")[:200]
                print(f"  [{i}] ({len(chk.page_content)} chars): {preview}...")

            preview_dir.mkdir(parents=True, exist_ok=True)
            for i, chk in enumerate(chunks):
                (preview_dir / f"chunk_{i:03}.txt").write_text(chk.page_content)

            for chk in chunks:
                chk.metadata["filename"] = str(path.relative_to(DOC_DIR)).replace("\\", "/")
                chk.metadata["chunk_id"] = chunk_id
                if "section" in chk.metadata:
                    chk.metadata["section_title"] = chk.metadata["section"]
                all_chunks_with_ids.append((chk, f"{file_hash}#{chunk_id}"))
                chunk_id += 1

        print(f"✅ {chunk_id} chunks prepared from {path.name}")

    print(f"\n📦 Total chunks to upload: {len(all_chunks_with_ids)}")
    for batch in batchify(all_chunks_with_ids):
        docs, ids = zip(*batch)
        vectorstore.add_documents(docs, ids=ids)
        print(f"↪️ Uploaded batch of {len(ids)}")

    print(f"\n🎉 All {len(all_chunks_with_ids)} chunks uploaded to '{INDEX_NAME}'!")

if __name__ == "__main__":
    main()
