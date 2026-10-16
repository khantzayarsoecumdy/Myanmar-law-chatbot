from pathlib import Path
from typing import Iterator
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

SUPPORTED = {".pdf", ".docx", ".txt"}

def read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".docx":
        doc = Document(str(path))
        blocks = []
        for child in doc.element.body.iterchildren():
            if child.tag == qn("w:p"):
                text = Paragraph(child, doc).text.strip()
                if text:
                    blocks.append(text)
            elif child.tag == qn("w:tbl"):
                table = Table(child, doc)
                for row in table.rows:
                    values = [cell.text.strip() for cell in row.cells]
                    line = " | ".join(value for value in values if value)
                    if line:
                        blocks.append(line)
        return "\n".join(blocks)
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[Page {i}]\n{text}")
        return "\n\n".join(pages)
    return ""

def split_text(text: str, chunk_size: int = 1000, overlap: int = 180) -> list[str]:
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind("။", start, end), text.rfind(". ", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        part = text[start:end].strip()
        if part:
            chunks.append(part)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks
