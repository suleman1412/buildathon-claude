from __future__ import annotations


def extract_text(pdf_path: str, max_chars: int = 20000) -> str:
    """Pull raw text out of a research paper PDF. v0 is text-only (no VLM),
    so figures and complex tables may be lost -- adequate for pulling repo
    URLs, dataset names, and headline metrics out of the body text."""
    import fitz  # PyMuPDF, lazy import

    doc = fitz.open(pdf_path)
    try:
        text = "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()
    return text[:max_chars]
