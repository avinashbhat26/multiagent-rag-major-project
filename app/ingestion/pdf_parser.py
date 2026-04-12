from io import BytesIO

from pypdf import PdfReader

from app.core.logging import get_logger

logger = get_logger(__name__)


class PDFParser:
    """Extracts page-level text content from PDF bytes."""

    def extract_pages(self, pdf_bytes: bytes) -> list[tuple[int, str]]:
        reader = PdfReader(BytesIO(pdf_bytes))
        pages: list[tuple[int, str]] = []

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if page_text:
                pages.append((page_number, " ".join(page_text.split())))

        if not pages:
            logger.warning("No extractable text found in PDF.")

        return pages
