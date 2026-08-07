# 📄 IMDU — Intelligent Document Understanding

Upload a PDF or an image, and ask questions about it in plain English.

IMDU reads your document — whether it's a clean digital PDF or a messy scanned page — understands its layout, and lets you chat with it. It combines **layout detection**, **OCR**, and **Retrieval-Augmented Generation (RAG)** with a large language model to give you answers grounded in the actual document.

> **Status:** MVP / learning project. Single-user, in-memory. Built to explore Document AI, OCR, and RAG end-to-end.

---

## ✨ What it does

- 📥 **Upload** a PDF or image (`.pdf`, `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.webp`)
- 🧠 **Understands** the document by picking the right path automatically:
  - **Digital PDFs** → text is extracted directly (fast, lossless)
  - **Scanned PDFs / images** → layout detection + OCR
- 🗂️ **Detects layout regions** (titles, paragraphs, tables, figures) so structure isn't lost
- 🔎 **Semantic search** over the document using vector embeddings + FAISS
- 💬 **Answers your questions** using Gemini, grounded in the retrieved text — with the source chunks shown so you can verify

---

## 🏗️ How it works

```text
                        Upload (PDF / image)
                                │
                     Is it a digital PDF with text?
                                │
              ┌─────────────────┴──────────────────┐
             YES                                    NO
              │                                     │
      Extract text directly              Render pages → images
        (PyMuPDF)                                   │
              │                        Classify type + language (Gemini)
              │                                     │
              │                        Detect layout (DocLayout-YOLO)
              │                                     │
              │                        OCR each region (PaddleOCR)
              └─────────────────┬──────────────────┘
                                │
                    Layout-aware chunks
                                │
              Embed chunks (Sentence-Transformers / MiniLM)
                                │
                     Index in FAISS (vector search)
                                │
                  ── When you ask a question ──
              Retrieve the most relevant chunks (top-k)
                                │
                 Answer with Gemini (grounded in context)
                                │
                        Answer + sources
```

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **Backend / API** | FastAPI, Uvicorn |
| **Document parsing** | PyMuPDF (fitz), Pillow |
| **Layout detection** | DocLayout-YOLO |
| **OCR** | PaddleOCR |
| **Embeddings** | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| **Vector search** | FAISS |
| **LLM** | Google Gemini (`gemini-2.5-flash`) |
| **Frontend** | HTML / CSS / JavaScript (single page) |
| **Language** | Python 3.12 |

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/IMDU.git
cd IMDU
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your Gemini API key

Create a `.env` file in the project root:

```env
GeminiAPIKey=your_google_gemini_api_key_here
```

> Get a free key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 4. Run the backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 5. Open the app

Visit **http://localhost:8000** in your browser, upload a document, and start asking questions.

API docs are auto-generated at **http://localhost:8000/docs**.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload a PDF/image and run the full pipeline |
| `POST` | `/ask` | Ask a question about the loaded document |
| `GET` | `/document/status` | Check current processing status |
| `GET` | `/health` | Health check |

**Example — ask a question:**

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main conclusion of this document?"}'
```

---

## 📁 Project Structure

```text
IMDU/
├── backend/
│   ├── main.py                  # FastAPI app + endpoints
│   └── pipeline/
│       ├── orchestrator.py      # Routes the file down the right path
│       ├── ingest.py            # Loads PDFs/images → page images
│       ├── text_extraction.py   # Digital-PDF text extraction (fast path)
│       ├── layout.py            # DocLayout-YOLO layout detection
│       ├── ocr.py               # PaddleOCR on each layout region
│       ├── metadata.py          # Gemini document type + language classifier
│       └── rag.py               # Chunking, embeddings, FAISS, Q&A
├── frontend/
│   └── index.html               # Upload + chat UI
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## 💡 Example Questions

Once a document is loaded, try asking:

- *What is the main objective of this document?*
- *Summarize the methodology.*
- *What datasets were used?*
- *List the key findings.*
- *What are the limitations discussed?*

If the answer isn't in the document, IMDU will tell you rather than make something up.

---

## ⚠️ Current Limitations

This is an MVP, and being upfront about scope:

- **Single-user / in-memory** — one document at a time; state isn't persisted
- **English OCR** — PaddleOCR is currently configured for English
- **No evaluation suite yet** — retrieval and OCR quality aren't benchmarked
- **Tables** are detected as regions but not parsed into structured tables
- The FAISS index is rebuilt per query (fine for demos, not for scale)

---

## 🗺️ Roadmap

- [ ] Persistent vector storage + per-document sessions (multi-user)
- [ ] Retrieval evaluation (recall@k) and OCR accuracy (CER/WER) benchmarks
- [ ] Hybrid search (BM25 + vector) and a reranker
- [ ] Structured table extraction
- [ ] Multilingual OCR wired to the detected language
- [ ] Background processing with progress updates
- [ ] Cloud deployment

---

## 📝 License

This project is for research, experimentation, and learning purposes.

---

<p align="center">Built with FastAPI · DocLayout-YOLO · PaddleOCR · Sentence-Transformers · FAISS · Gemini</p>
