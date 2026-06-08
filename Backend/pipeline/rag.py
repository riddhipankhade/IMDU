from typing import List
import google.generativeai as genai


def document_to_text(result: dict) -> str:
    """
    Convert OCR output into one large text document.
    """

    text = ""

    for page in result["pages"]:

        for block in page["blocks"]:

            block_text = block.get("text", "").strip()

            if block_text:
                text += block_text + "\n\n"

    return text


def chunk_text(
    text: str,
    chunk_size: int = 1000
) -> List[str]:
    """
    Split document text into chunks.

    Example:
        5000 chars
        ->
        5 chunks of ~1000 chars
    """

    chunks = []

    for i in range(0, len(text), chunk_size):

        chunk = text[i:i + chunk_size]

        chunks.append(chunk)

    return chunks


def ask_document(
    question: str,
    chunks: List[str]
) -> str:
    """
    Ask Gemini a question about the document.

    Current version:
    Uses all chunks.

    Future version:
    Will retrieve only relevant chunks.
    """

    document_context = "\n\n".join(chunks)

    prompt = f"""
You are answering questions about a document.

Use ONLY the information provided.

DOCUMENT:

{document_context}

QUESTION:
{question}

ANSWER:
"""

    model = genai.GenerativeModel(
        "gemini-2.5-flash"
    )

    response = model.generate_content(prompt)

    return response.text