"""
Ingest rubric + reference material into Pinecone for a given assignment.
Run this whenever a teacher creates/updates an assignment's grading material.

Usage:
  python ingest.py <assignment_id> <file_path> [doc_type]
  doc_type: rubric | reference | exemplar   (default: rubric)
"""

import os
import sys

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "assignments")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

pc = Pinecone(api_key=PINECONE_API_KEY)


def ensure_index():
    existing = [i["name"] for i in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,  # text-embedding-3-small
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


def ingest_assignment_material(assignment_id: str, text: str, doc_type: str = "rubric"):
    ensure_index()
    index = pc.Index(PINECONE_INDEX_NAME)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)

    vectorstore = PineconeVectorStore(
        index=index, embedding=embeddings, namespace=f"assignment_{assignment_id}"
    )
    vectorstore.add_texts(
        chunks,
        metadatas=[{"doc_type": doc_type, "assignment_id": assignment_id} for _ in chunks],
    )
    print(f"Ingested {len(chunks)} chunks for assignment {assignment_id} ({doc_type})")


if __name__ == "__main__":
    assignment_id, file_path = sys.argv[1], sys.argv[2]
    doc_type = sys.argv[3] if len(sys.argv) > 3 else "rubric"
    with open(file_path, "r") as f:
        text = f.read()
    ingest_assignment_material(assignment_id, text, doc_type)
