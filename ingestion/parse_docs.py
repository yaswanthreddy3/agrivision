import os
import re
import signal
from pathlib import Path
from dataclasses import dataclass
import pdfplumber
from pypdf import PdfReader
import logfire
from config.observability import setup_logfire

setup_logfire()

RAW_PDF_DIR = "data/docs/niphm/"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
TIMEOUT_SECONDS = 30


@dataclass
class DocChunk:
    text: str
    source_file: str
    chunk_index: int
    page_number: int


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException()


def simple_recursive_split(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Lightweight recursive splitter — no LangChain dependency needed for this."""
    separators = ["\n\n", "\n", ". ", " "]

    def split_by_sep(text: str, seps: list[str]) -> list[str]:
        if not seps:
            return [text]
        sep = seps[0]
        parts = text.split(sep)
        return [p + sep for p in parts[:-1]] + [parts[-1]] if len(parts) > 1 else split_by_sep(text, seps[1:])

    chunks = []
    current = ""
    pieces = split_by_sep(text, separators)

    for piece in pieces:
        if len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            if current:
                chunks.append(current.strip())
            current = current[-overlap:] + piece if overlap > 0 else piece

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if c]


def extract_text_pdfplumber(pdf_path: str) -> list[tuple[int, str]]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append((i + 1, text))
    return pages


def extract_text_pypdf(pdf_path: str) -> list[tuple[int, str]]:
    pages = []
    reader = PdfReader(pdf_path)
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append((i + 1, text))
    return pages


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(pdf_path: str) -> list[tuple[int, str]]:
    with logfire.span("extract_pdf", pdf_path=pdf_path):
        pages = extract_text_pdfplumber(pdf_path)

        if not pages:
            logfire.info("pdfplumber_empty_trying_pypdf", pdf_path=pdf_path)
            pages = extract_text_pypdf(pdf_path)

        if not pages:
            logfire.info("extraction_failed_likely_scanned", pdf_path=pdf_path)
            return []

        logfire.info("extraction_success", pdf_path=pdf_path, num_pages=len(pages))
        return pages


def chunk_document(pages: list[tuple[int, str]], source_file: str) -> list[DocChunk]:
    chunks = []
    chunk_idx = 0
    for page_num, page_text in pages:
        cleaned = clean_text(page_text)
        if len(cleaned) < 30:
            continue

        page_chunks = simple_recursive_split(cleaned)
        for chunk_text in page_chunks:
            chunks.append(DocChunk(
                text=chunk_text,
                source_file=source_file,
                chunk_index=chunk_idx,
                page_number=page_num
            ))
            chunk_idx += 1

    return chunks


def process_all_pdfs(raw_dir: str = RAW_PDF_DIR) -> list[DocChunk]:
    all_chunks = []
    pdf_files = list(Path(raw_dir).glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {raw_dir}")
        return []

    print(f"Found {len(pdf_files)} PDF(s) in {raw_dir}\n")

    for idx, pdf_path in enumerate(pdf_files, 1):
        size_kb = pdf_path.stat().st_size / 1024
        print(f"[{idx}/{len(pdf_files)}] Processing: {pdf_path.name} ({size_kb:.0f} KB)...", flush=True)

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIMEOUT_SECONDS)

        try:
            pages = extract_pdf(str(pdf_path))
        except TimeoutException:
            print(f"    ⏱️  TIMEOUT after {TIMEOUT_SECONDS}s — skipping")
            signal.alarm(0)
            continue
        except Exception as e:
            print(f"    ❌ ERROR: {e}")
            signal.alarm(0)
            continue
        finally:
            signal.alarm(0)

        if not pages:
            print(f"    ⚠️  SKIPPED (no extractable text — likely scanned)")
            continue

        chunks = chunk_document(pages, source_file=pdf_path.name)
        all_chunks.extend(chunks)
        print(f"    ✅ {len(pages)} pages → {len(chunks)} chunks")

    return all_chunks


if __name__ == "__main__":
    chunks = process_all_pdfs()
    print(f"\n--- Summary ---")
    print(f"Total chunks created: {len(chunks)}")
    if chunks:
        print(f"\nSample chunk (first one):")
        print(f"Source: {chunks[0].source_file}, Page: {chunks[0].page_number}")
        print(f"Text: {chunks[0].text[:300]}...")