import os
import json
from typing import List
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
import re
load_dotenv()



# ---------- setup Gemini ----------
api_key = os.getenv("GeminiAPIKey")

if not api_key:
    raise ValueError("GeminiAPIKey not found in environment")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")
# ---------------------------------


def classify_document(pages: List[Image.Image]) -> dict:
    """
    Classify document type and language using Gemini
    """

    if not pages:
        return {
            "document_type": "other",
            "language": "unknown",
            "pages": 0
        }

    first_page = pages[0]

    prompt = """
Return ONLY valid JSON. Do NOT include explanation or markdown.

{
  "document_type": "invoice | resume | report | research paper | letter | textbook | other",
  "language": "language name"
}
"""

    try:
        response = model.generate_content([prompt, first_page])
        raw = response.text

        match = re.search(r"\{.*\}", raw, re.DOTALL)

        if match:
            result = json.loads(match.group(0))
        else:
            result = {}

    except Exception as e:
        print("Gemini error:", e)
        result = {}

    doc_type = result.get("document_type", "other").lower()
    language = result.get("language", "unknown").lower()

    return {
        "document_type": doc_type,
        "language": language,
        "pages": len(pages)
}