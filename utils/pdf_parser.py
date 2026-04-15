"""
utils/pdf_parser.py - PDF text extraction using pdfplumber.

pdfplumber is used here for its pure Python dependency tree,
which ensures compatibility across all Windows environments without
requiring a Visual Studio C++ compiler.
"""

from typing import List
import pdfplumber

from utils.logger import get_logger

logger = get_logger(__name__)


def extract_text_from_pdf(filepath: str) -> str:
    """
    Extracts and concatenates text from all pages of a PDF file.

    Args:
        filepath: Absolute or relative path to the PDF file.

    Returns:
        A single string containing all extracted text, with pages
        separated by newlines.

    Raises:
        RuntimeError: If the file cannot be opened or parsed.
    """
    try:
        pages_text: List[str] = []
        
        with pdfplumber.open(filepath) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    pages_text.append(page_text)
                    logger.debug(f"PDF page {page_num + 1}: {len(page_text)} chars extracted.")
                else:
                    logger.warning(
                        f"PDF page {page_num + 1} of '{filepath}' has no extractable text. "
                        "May be an image-only page."
                    )
                    
        full_text = "\n".join(pages_text)
        logger.info(f"PDF parsed: '{filepath}' — {len(pages_text)} pages, "
                    f"{len(full_text)} total characters.")
        return full_text

    except Exception as e:
        logger.error(f"Failed to parse PDF '{filepath}': {e}")
        raise RuntimeError(f"PDF parsing failed for '{filepath}': {e}") from e


def extract_text_from_txt(filepath: str) -> str:
    """
    Reads raw text from a .txt file, trying UTF-8 then latin-1 encoding.

    Args:
        filepath: Path to the text file.

    Returns:
        Full file contents as a string.
    """
    for encoding in ("utf-8", "latin-1"):
        try:
            with open(filepath, "r", encoding=encoding) as f:
                text = f.read()
            logger.info(f"TXT parsed: '{filepath}' — {len(text)} characters [{encoding}].")
            return text
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"Failed to read TXT '{filepath}': {e}")
            raise RuntimeError(f"TXT read failed for '{filepath}': {e}") from e

    raise RuntimeError(f"Could not decode '{filepath}' with utf-8 or latin-1.")


def extract_text(filepath: str) -> str:
    """
    Dispatcher: routes to the correct parser based on file extension.

    Supported extensions: .pdf, .txt
    """
    lower = filepath.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(filepath)
    elif lower.endswith(".txt"):
        return extract_text_from_txt(filepath)
    else:
        raise ValueError(f"Unsupported file type: '{filepath}'. Only PDF and TXT are accepted.")
