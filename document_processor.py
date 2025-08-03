import requests
import pdfplumber
import hashlib
import re
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import time
import logging
from config import Config

# Handle different Pinecone versions
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_V3 = True
except ImportError:
    try:
        import pinecone
        PINECONE_V3 = False
    except ImportError:
        raise ImportError("Pinecone is not installed. Install with: pip install pinecone-client")

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        
        # Initialize Pinecone based on version
        if PINECONE_V3:
            self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        else:
            self.pc = None
        
        self._ensure_index()
    
    def _ensure_index(self):
        """Ensure Pinecone index exists."""
        try:
            if PINECONE_V3:
                if self.config.PINECONE_INDEX_NAME not in self.pc.list_indexes().names():
                    self.pc.create_index(
                        name=self.config.PINECONE_INDEX_NAME,
                        dimension=self.config.EMBEDDING_DIMENSION,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )
                    while not self.pc.describe_index(self.config.PINECONE_INDEX_NAME).status["ready"]:
                        time.sleep(1)
                self.index = self.pc.Index(self.config.PINECONE_INDEX_NAME)
            else:
                pinecone.init(api_key=self.config.PINECONE_API_KEY, environment="us-east-1-aws")
                if self.config.PINECONE_INDEX_NAME not in pinecone.list_indexes():
                    pinecone.create_index(
                        name=self.config.PINECONE_INDEX_NAME,
                        dimension=self.config.EMBEDDING_DIMENSION,
                        metric="cosine"
                    )
                    while not pinecone.describe_index(self.config.PINECONE_INDEX_NAME).status["ready"]:
                        time.sleep(1)
                self.index = pinecone.Index(self.config.PINECONE_INDEX_NAME)
        except Exception as e:
            logger.error(f"Error setting up Pinecone index: {e}")
            raise
    
    def process_document(self, document_url: str) -> str:
        """Download, process document and store embeddings."""
        document_id = hashlib.md5(document_url.encode()).hexdigest()[:12]
        
        if self._is_document_processed(document_id):
            logger.info(f"Document {document_id} already processed")
            return document_id
        
        # Download and process
        pdf_path = self._download_document(document_url)
        chunks = self._extract_content(pdf_path)
        self._store_embeddings(chunks, document_id)
        
        logger.info(f"Document {document_id} processed: {len(chunks)} chunks")
        return document_id
    
    def _download_document(self, url: str) -> str:
        """Download document from URL."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            filename = f"temp_doc_{int(time.time())}.pdf"
            with open(filename, "wb") as f:
                f.write(response.content)
            
            return filename
        except Exception as e:
            logger.error(f"Error downloading document: {e}")
            raise
    
    def _is_document_processed(self, document_id: str) -> bool:
        """Check if document is already processed."""
        try:
            dummy_vector = [0.0] * self.config.EMBEDDING_DIMENSION
            results = self.index.query(
                vector=dummy_vector,
                filter={"document_id": document_id},
                top_k=1,
                include_metadata=True
            )
            return len(results.get("matches", [])) > 0
        except:
            return False
    
    def _extract_content(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract content with smart chunking."""
        chunks = []
        
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            
            # Extract text and tables
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text() or ""
                full_text += f"\n[PAGE {page_num}]\n" + page_text
                
                # Extract meaningful tables
                tables = page.extract_tables()
                if tables:
                    for table_idx, table in enumerate(tables):
                        if self._is_meaningful_table(table):
                            table_text = self._format_table(table, page_text[:200])
                            chunks.append({
                                "text": table_text,
                                "page": page_num,
                                "type": "table",
                                "chunk_id": f"table_{page_num}_{table_idx}"
                            })
            
            # Create text chunks
            text_chunks = self._create_chunks(full_text)
            chunks.extend(text_chunks)
        
        return chunks
    
    def _is_meaningful_table(self, table: List[List]) -> bool:
        """Check if table contains meaningful data."""
        if not table or len(table) < 2:
            return False
        
        table_text = " ".join([" ".join([str(cell) for cell in row if cell]) for row in table]).lower()
        keywords = ["premium", "coverage", "benefit", "limit", "amount", "period", "plan", "sum"]
        
        return any(keyword in table_text for keyword in keywords)
    
    def _format_table(self, table: List[List], context: str = "") -> str:
        """Format table with context."""
        if not table:
            return ""
        
        formatted = f"Context: {context}\n\nTable:\n" if context else "Table:\n"
        
        for row in table[:15]:  # Limit rows
            clean_row = [str(cell).strip() if cell else "" for cell in row]
            formatted += " | ".join(clean_row) + "\n"
        
        return formatted
    
    def _create_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Create text chunks with overlap."""
        # Remove page markers for chunking
        clean_text = re.sub(r'\[PAGE \d+\]', '', text)
        
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', clean_text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.config.CHUNK_SIZE and current_chunk:
                chunk_text = ' '.join(current_chunk)
                page_num = self._extract_page_number(chunk_text, text)
                
                chunks.append({
                    "text": chunk_text,
                    "page": page_num,
                    "type": "text",
                    "chunk_id": f"text_{chunk_id}"
                })
                
                chunk_id += 1
                # Overlap with last sentence
                current_chunk = [current_chunk[-1], sentence] if current_chunk else [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            page_num = self._extract_page_number(chunk_text, text)
            chunks.append({
                "text": chunk_text,
                "page": page_num,
                "type": "text",
                "chunk_id": f"text_{chunk_id}"
            })
        
        return chunks
    
    def _extract_page_number(self, chunk_text: str, full_text: str) -> int:
        """Extract page number for a chunk."""
        # Find chunk position in full text
        chunk_pos = full_text.find(chunk_text[:100])
        if chunk_pos == -1:
            return 1
        
        # Count page markers before this position
        text_before = full_text[:chunk_pos]
        page_matches = re.findall(r'\[PAGE (\d+)\]', text_before)
        
        return int(page_matches[-1]) if page_matches else 1
    
    def _store_embeddings(self, chunks: List[Dict[str, Any]], document_id: str):
        """Store embeddings in Pinecone."""
        batch_size = 50
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            vectors = []
            
            for j, chunk in enumerate(batch):
                text = chunk["text"]
                if not text.strip():
                    continue
                
                embedding = self.embedder.encode(text).tolist()
                vector_id = f"{document_id}_{i + j}"
                
                metadata = {
                    "document_id": document_id,
                    "text": text[:1000],  # Truncate for metadata storage
                    "page": chunk["page"],
                    "type": chunk["type"],
                    "chunk_id": chunk["chunk_id"]
                }
                
                vectors.append((vector_id, embedding, metadata))
            
            if vectors:
                self.index.upsert(vectors)
                logger.info(f"Stored batch {i//batch_size + 1}: {len(vectors)} vectors")