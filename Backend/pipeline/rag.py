"""
rag.py
------
RAG (Retrieval-Augmented Generation) pipeline.

Steps:
1. create_layout_chunks()  — split OCR/text output into small chunks with metadata
2. create_embeddings()     — embed chunks using sentence-transformers
3. build_faiss_index()     — build in-memory FAISS index for fast similarity search
4. retrieve_chunks()       — find top-k most relevant chunks for a question
5. ask_document()          — full end-to-end: embed question → retrieve → ask Gemini

The FAISS index is rebuilt per query session (stateless, in-memory).
For production you'd persist the index to disk — kept simple here for clarity.
"""

import os
import faiss
import numpy as np

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

load_dotenv()

# ── Gemini setup ────────────────────────────────────────────────────────────
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

genai.configure(api_key=api_key)
_llm = genai.GenerativeModel("gemini-2.5-flash")

# ── Embedding model (loaded once) ───────────────────────────────────────────
# all-MiniLM-L6-v2: fast, lightweight, 384-dim. Good for semantic search.
_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ── 1. Chunking ─────────────────────────────────────────────────────────────

def create_layout_chunks(document_json: dict, max_chunk_size: int = 1000) -> list:
    """
    Convert the OCR/text output into chunks.

    Each chunk is a dict:
        {"page": int, "type": str, "text": str}

    Large blocks (>max_chunk_size chars) are split into overlapping sub-chunks.
    Overlap of 100 chars prevents cutting a sentence mid-thought.
    """
    chunks = []
    overlap = 100  # chars to carry over between sub-chunks

    for page in document_json["pages"]:
        page_num = page["page"]

        for block in page["blocks"]:
            text = block["text"].strip()
            if not text:
                continue

            label = block.get("label", "text")

            if len(text) <= max_chunk_size:
                chunks.append({"page": page_num, "type": label, "text": text})
            else:
                # Sliding window split with overlap
                start = 0
                while start < len(text):
                    end = start + max_chunk_size
                    chunks.append({
                        "page": page_num,
                        "type": label,
                        "text": text[start:end],
                    })
                    start += max_chunk_size - overlap

    return chunks


def create_chunks_from_text(text: str, max_chunk_size: int = 1000) -> list:
    """
    Fallback chunker for digital PDFs (no layout blocks available).
    Splits on paragraphs first, then falls back to sliding window.
    """
    chunks = []
    overlap = 100
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    buffer = ""
    for para in paragraphs:
        if len(buffer) + len(para) < max_chunk_size:
            buffer += para + "\n\n"
        else:
            if buffer:
                chunks.append({"page": 1, "type": "digital_text", "text": buffer.strip()})
            buffer = para + "\n\n"
    if buffer:
        chunks.append({"page": 1, "type": "digital_text", "text": buffer.strip()})

    return chunks


# ── 2. Embeddings ────────────────────────────────────────────────────────────

def create_embeddings(chunks: list) -> np.ndarray:
    """
    Encode chunk texts into float32 numpy array.
    Shape: (num_chunks, 384) for all-MiniLM-L6-v2.
    """
    texts = [chunk["text"] for chunk in chunks]
    embeddings = _embedding_model.encode(texts, convert_to_numpy=True)
    return embeddings.astype(np.float32)


# ── 3. FAISS Index ───────────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Build an L2 (Euclidean distance) flat index.
    IndexFlatL2 does exact nearest-neighbour search — fine for <100k chunks.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


# ── 4. Retrieval ─────────────────────────────────────────────────────────────

def retrieve_chunks(
    question: str,
    chunks: list,
    index: faiss.IndexFlatL2,
    top_k: int = 5,
) -> list:
    """
    Embed the question, search FAISS, return the top_k most similar chunks.
    """
    q_emb = _embedding_model.encode([question], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(q_emb, top_k)

    retrieved = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):  # guard against -1 (empty index edge case)
            retrieved.append(chunks[idx])
    return retrieved


# ── 5. End-to-end QA ─────────────────────────────────────────────────────────

def ask_document(question: str, chunks: list) -> dict:
    """
    Full RAG pipeline:
        question + chunks → embed → retrieve → build prompt → Gemini → answer

    Returns:
        {
            "answer": str,
            "source_chunks": list   ← which chunks were used
        }
    """
    if not chunks:
        return {
            "answer": "No document content available to answer from.",
            "source_chunks": [],
        }

    embeddings = create_embeddings(chunks)
    index = build_faiss_index(embeddings)
    retrieved = retrieve_chunks(question, chunks, index)

    context = ""
    for chunk in retrieved:
        context += (
            f"[Page {chunk['page']} | Type: {chunk['type']}]\n"
            f"{chunk['text']}\n\n"
        )

    prompt = f"""You are a helpful document assistant.
Answer the question ONLY using the provided context below.
If the answer is not in the context, say: "I could not find that information in the document."
Do not make up information.

--- CONTEXT ---
{context}
--- END CONTEXT ---

Question: {question}

Answer:"""

    response = _llm.generate_content(prompt)

    return {
        "answer": response.text.strip(),
        "source_chunks": retrieved,
    }
