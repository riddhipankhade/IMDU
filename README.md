# IMDU – Intelligent Multilingual Document Understanding

An AI-powered document understanding system that transforms PDFs, scanned documents, research papers, reports, forms, and images into searchable knowledge using OCR, layout analysis, retrieval-augmented generation (RAG), and large language models.

## Overview

Modern documents contain far more than plain text. Research papers, invoices, contracts, reports, and forms often include:

* Complex layouts
* Tables
* Figures
* Captions
* Multi-column structures
* Scanned content without embedded text

Traditional document parsing pipelines frequently lose structural information, resulting in poor search and question-answering performance.

IMDU addresses this problem by combining:

* Document ingestion
* Layout-aware document understanding
* OCR-based text extraction
* Semantic chunking
* Vector search
* Retrieval-Augmented Generation (RAG)
* Gemini-powered document question answering

The system allows users to upload a document and ask natural language questions while preserving document structure and context.

---

## Key Features

### Document Processing

* PDF ingestion
* Image ingestion
* Multi-page document support
* Digital PDF text extraction
* OCR pipeline for scanned documents

### Document Understanding

* Layout detection using DocLayout-YOLO
* Block-level understanding
* Detection of:

  * Titles
  * Plain text
  * Tables
  * Figure captions
  * Figures
  * Other document regions

### OCR

* PaddleOCR-based text extraction
* Region-wise OCR
* Layout-aware extraction
* Automatic handling of scanned documents

### Retrieval-Augmented Generation (RAG)

* Layout-aware chunk generation
* Semantic embeddings
* FAISS vector indexing
* Similarity-based retrieval
* Context-aware question answering

### AI-Powered Question Answering

* Gemini integration
* Retrieval-grounded responses
* Source-aware context generation
* Reduced hallucination through document retrieval

### Web Application

* FastAPI backend
* Interactive frontend
* Document upload workflow
* Natural language querying

---

## System Architecture

```text
User Upload
      │
      ▼
Document Ingestion
      │
      ├──────────────► Digital PDF
      │                     │
      │                     ▼
      │             Direct Text Extraction
      │
      ▼
Scanned PDF / Image
      │
      ▼
Layout Detection
(DocLayout-YOLO)
      │
      ▼
Region-Based OCR
(PaddleOCR)
      │
      ▼
Layout-Aware Chunks
      │
      ▼
Embedding Generation
(Sentence Transformers)
      │
      ▼
FAISS Vector Index
      │
      ▼
Semantic Retrieval
      │
      ▼
Gemini LLM
      │
      ▼
Answer Generation
```

---

## Technology Stack

### AI & Machine Learning

* PaddleOCR
* DocLayout-YOLO
* Sentence Transformers
* FAISS
* Google Gemini

### Backend

* FastAPI
* Python

### Document Processing

* PyMuPDF
* Pillow

### Vector Search

* FAISS

### Frontend

* HTML
* CSS
* JavaScript

---

## Project Structure

```text
IMDU/
│
├── Backend/
│   ├── pipeline/
│   │   ├── ingest.py
│   │   ├── text_extraction.py
│   │   ├── layout.py
│   │   ├── ocr.py
│   │   ├── metadata.py
│   │   ├── rag.py
│   │   └── orchestrator.py
│   │
│   └── main.py
│
├── frontend/
│   └── index.html
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Current MVP Capabilities

Implemented:

* Document upload
* PDF processing
* Digital PDF text extraction
* OCR for scanned documents
* Layout detection
* Layout-aware chunking
* Embedding generation
* FAISS retrieval
* Gemini-based document QA
* FastAPI backend
* Frontend interface

---

## Example Questions

After uploading a document, users can ask:

* What is the main objective of this paper?
* Summarize the methodology.
* What datasets were used?
* Explain the results section.
* What are the key findings?
* List limitations discussed by the authors.
* What does Table 2 show?

---

## Future Roadmap

### Retrieval Improvements

* Persistent vector storage
* Hybrid Search (BM25 + Vector Search)
* Reranking models
* Citation generation

### Document Understanding

* Table extraction
* Form understanding
* Structured metadata extraction
* Knowledge graph generation

### Scalability

* Multi-document support
* User authentication
* Document database
* Cloud deployment

### Advanced AI

* Agentic RAG
* GraphRAG
* Multi-agent workflows
* Research assistant mode

---

## Learning Objectives

This project was built to explore:

* Document AI
* OCR systems
* Retrieval-Augmented Generation
* Vector databases
* Information retrieval
* FastAPI backend development
* LLM integration
* End-to-end AI system design

---

## Disclaimer

This project is currently an MVP (Minimum Viable Product) intended for research, experimentation, and learning purposes. Future versions will focus on retrieval quality, scalability, and advanced document understanding capabilities.
