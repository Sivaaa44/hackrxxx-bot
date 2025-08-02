# import requests
# import pdfplumber
# import spacy
# import tiktoken
# from sentence_transformers import SentenceTransformer
# from pinecone import Pinecone
# from typing import List, Dict, Any
# from fastapi import FastAPI, Header

# # Initialize tools
# nlp = spacy.load("en_core_web_sm")
# tokenizer = tiktoken.get_encoding("cl100k_base")

# # Initialize Pinecone with provided API key
# pc = Pinecone(api_key="pcsk_2XxfGa_SEg9J4q8YAijGaCEGdvBisXv4ioR7yYsjuZLZmbxFriddYvW7vg9RzhskvaDRLy")
# index = pc.Index("hackrxxx")

# # Use llama-text-embed-v2 (simulated here; replace with actual API if available)
# # Note: sentence-transformers doesn't natively support llama-text-embed-v2. We'll use a compatible model and adjust later if needed.
# embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Placeholder; replace with llama-text-embed-v2 if accessible via API

# app = FastAPI()

# def fetch_data(url: str, output_path: str = "policy.pdf") -> None:
#     """Download PDF from URL."""
#     response = requests.get(url)
#     with open(output_path, "wb") as f:
#         f.write(response.content)

# def extract_text(page) -> str:
#     """Extract text from a PDF page."""
#     return page.extract_text() or ""

# def extract_tables(page) -> List[List[List[str]]]:
#     """Extract and clean tables from a PDF page."""
#     tables = page.extract_tables()
#     if not tables:
#         return []
    
#     cleaned_tables = []
#     for table in tables:
#         if not table or not any(any(cell for cell in row) for row in table):
#             continue
#         cleaned_table = [[str(cell).strip() if cell is not None else "" for cell in row] for row in table]
#         cleaned_tables.append(cleaned_table)
#     return cleaned_tables

# def chunk_text(text: str, max_tokens: int = 1024, overlap: int = 50) -> List[str]:
#     """Split text into chunks with overlap, preserving semantic boundaries."""
#     doc = nlp(text)
#     chunks = []
#     current_chunk = []
#     current_tokens = 0
    
#     for sent in doc.sents:
#         sent_tokens = len(tokenizer.encode(sent.text))
#         if current_tokens + sent_tokens > max_tokens:
#             chunks.append(" ".join(current_chunk))
#             current_chunk = current_chunk[-overlap:]  # Keep overlap
#             current_tokens = sum(len(tokenizer.encode(s)) for s in current_chunk)
#         current_chunk.append(sent.text)
#         current_tokens += sent_tokens
    
#     if current_chunk:
#         chunks.append(" ".join(current_chunk))
    
#     return chunks

# def process_and_store_chunks(chunks: List[Dict[str, Any]], document_id: str) -> None:
#     """Generate embeddings and store in Pinecone."""
#     for i, chunk in enumerate(chunks):
#         text = chunk["text"]
#         if chunk["metadata"]["type"] == "table":
#             sub_chunks = [text]  # treat the whole table as one chunk
#         else:
#             sub_chunks = chunk_text(text)  # split text into sub-chunks
#         for j, sub_chunk in enumerate(sub_chunks):
#             embedding = embedder.encode(sub_chunk).tolist()
#             vector_id = f"{document_id}_chunk_{i}_{j}"
#             metadata = chunk["metadata"].copy()
#             metadata["chunk_index"] = j
#             # Store the text in metadata for retrieval
#             metadata["text"] = sub_chunk
#             index.upsert([(vector_id, embedding, metadata)])
#             print(f"Stored chunk {vector_id}")

# def extract_pdf_content(pdf_path: str) -> List[Dict[str, Any]]:
#     """Extract text and tables from PDF, prioritizing tables."""
#     pages_data = []
#     with pdfplumber.open(pdf_path) as pdf:
#         for page_num, page in enumerate(pdf.pages, 1):
#             page_data = {"page_number": page_num, "chunks": []}
            
#             # Extract tables first
#             tables = extract_tables(page)
#             if tables:
#                 for table_idx, table in enumerate(tables):
#                     table_text = "\n".join(" | ".join(row) for row in table)
#                     page_data["chunks"].append({
#                         "text": table_text,
#                         "metadata": {
#                             "document": pdf_path,
#                             "page": page_num,
#                             "type": "table",
#                             "table_id": f"table_{page_num}_{table_idx}"
#                         }
#                     })
#             else:
#                 # Extract text if no tables
#                 page_text = extract_text(page)
#                 if page_text:
#                     text_chunks = chunk_text(page_text)
#                     for chunk_idx, chunk in enumerate(text_chunks):
#                         page_data["chunks"].append({
#                             "text": chunk,
#                             "metadata": {
#                                 "document": pdf_path,
#                                 "page": page_num,
#                                 "type": "text",
#                                 "chunk_id": f"text_{page_num}_{chunk_idx}"
#                             }
#                         })
            
#             pages_data.append(page_data)
    
#     return pages_data

# def save_extracted_data(pages_data: List[Dict[str, Any]], output_file: str = "extracted_pdf_data.txt") -> None:
#     """Save extracted data in a readable format."""
#     with open(output_file, "w", encoding="utf-8") as f:
#         for page_data in pages_data:
#             f.write(f"\n{'='*60}\n")
#             f.write(f"PAGE {page_data['page_number']}\n")
#             f.write(f"{'='*60}\n\n")
#             for chunk in page_data["chunks"]:
#                 f.write(f"TYPE: {chunk['metadata']['type']}\n")
#                 f.write(f"TEXT: {chunk['text']}\n\n")
    
#     print(f"PDF Extraction Complete!")
#     print(f"- Pages processed: {len(pages_data)}")
#     print(f"- Chunks extracted: {sum(len(page['chunks']) for page in pages_data)}")
#     print(f"- Output saved to: {output_file}")

# def process_query(query: str) -> Dict[str, Any]:
#     """Process a query and retrieve relevant chunks."""
#     query_embedding = embedder.encode(query).tolist()
#     results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    
#     top_chunk = results["matches"][0]
#     # Extract text from metadata
#     return {
#         "query": query,
#         "answer": top_chunk["metadata"]["text"],
#         "source": top_chunk["metadata"],
#         "confidence": top_chunk["score"]
#     }

# # @app.post("/hackrx/run")
# # async def run_submission(data: Dict, authorization: str = Header(...)):
# #     """FastAPI endpoint for query processing."""
# #     if authorization != "Bearer 820dac0a840355cdf470fe4cae4a85dc938fe856b168ab47aeb306824aaccef8":
# #         return {"error": "Invalid token"}
    
# #     fetch_data(data["documents"])
# #     pages_data = extract_pdf_content("policy.pdf")
# #     all_chunks = [chunk for page in pages_data for chunk in page["chunks"]]
# #     process_and_store_chunks(all_chunks, "arogya_sanjeevani_policy")
    
# #     answers = [process_query(q) for q in data["questions"]]
# #     return {"answers": answers}

# def main():
#     pdf_url = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"
#     fetch_data(pdf_url)
    
#     pages_data = extract_pdf_content("policy.pdf")
#     all_chunks = [chunk for page in pages_data for chunk in page["chunks"]]
#     process_and_store_chunks(all_chunks, "arogya_sanjeevani_policy")
    
#     save_extracted_data(pages_data)
    
#     sample_query = "What is the waiting period for cataract surgery and the benifits of cataract surgery?"
#     result = process_query(sample_query)
#     print("Sample query result:", result)

# if __name__ == "__main__":
#     main()