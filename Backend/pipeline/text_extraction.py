import fitz


def has_text_layer(pdf_path: str) -> bool:

    doc = fitz.open(pdf_path)

    for page in doc:

        text = page.get_text().strip()

        if len(text) > 50:
            return True

    return False


def extract_pdf_text(pdf_path: str) -> str:

    doc = fitz.open(pdf_path)

    full_text = ""

    for page in doc:

        full_text += page.get_text()
        full_text += "\n\n"

    return full_text