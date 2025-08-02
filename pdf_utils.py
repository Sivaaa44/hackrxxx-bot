import requests
import pdfplumber
import spacy
import tiktoken
from typing import List, Dict, Any

nlp = spacy.load("en_core_web_sm")
tokenizer = tiktoken.get_encoding("cl100k_base")

def fetch_data(url: str, output_path: str = "policy.pdf") -> None:
    response = requests.get(url)
    with open(output_path, "wb") as f:
        f.write(response.content)

def extract_text(page) -> str:
    return page.extract_text() or ""

def extract_tables(page) -> List[List[List[str]]]:
    tables = page.extract_tables()
    if not tables:
        return []
    cleaned_tables = []
    for table in tables:
        if not table or not any(any(cell for cell in row) for row in table):
            continue
        cleaned_table = [[str(cell).strip() if cell is not None else "" for cell in row] for row in table]
        cleaned_tables.append(cleaned_table)
    return cleaned_tables

def chunk_text(text: str, max_tokens: int = 1024, overlap: int = 50) -> List[str]:
    doc = nlp(text)
    chunks = []
    current_chunk = []
    current_tokens = 0
    for sent in doc.sents:
        sent_tokens = len(tokenizer.encode(sent.text))
        if current_tokens + sent_tokens > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:]
            current_tokens = sum(len(tokenizer.encode(s)) for s in current_chunk)
        current_chunk.append(sent.text)
        current_tokens += sent_tokens
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def extract_pdf_content(pdf_path: str) -> List[Dict[str, Any]]:
    pages_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            page_data = {"page_number": page_num, "chunks": []}
            tables = extract_tables(page)
            if tables:
                for table_idx, table in enumerate(tables):
                    table_text = "\n".join(" | ".join(row) for row in table)
                    page_data["chunks"].append({
                        "text": table_text,
                        "metadata": {
                            "document": pdf_path,
                            "page": page_num,
                            "type": "table",
                            "table_id": f"table_{page_num}_{table_idx}"
                        }
                    })
            else:
                page_text = extract_text(page)
                if page_text:
                    text_chunks = chunk_text(page_text)
                    for chunk_idx, chunk in enumerate(text_chunks):
                        page_data["chunks"].append({
                            "text": chunk,
                            "metadata": {
                                "document": pdf_path,
                                "page": page_num,
                                "type": "text",
                                "chunk_id": f"text_{page_num}_{chunk_idx}"
                            }
                        })
            pages_data.append(page_data)
    return pages_data

def save_extracted_data(pages_data: List[Dict[str, Any]], output_file: str = "extracted_pdf_data.txt") -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        for page_data in pages_data:
            f.write(f"\n{'='*60}\n")
            f.write(f"PAGE {page_data['page_number']}\n")
            f.write(f"{'='*60}\n\n")
            for chunk in page_data["chunks"]:
                f.write(f"TYPE: {chunk['metadata']['type']}\n")
                f.write(f"TEXT: {chunk['text']}\n\n")
    print(f"PDF Extraction Complete!")
    print(f"- Pages processed: {len(pages_data)}")
    print(f"- Chunks extracted: {sum(len(page['chunks']) for page in pages_data)}")
    print(f"- Output saved to: {output_file}")