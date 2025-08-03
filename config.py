import os
from dataclasses import dataclass

@dataclass
class Config:
    PINECONE_API_KEY: str = "pcsk_2XxfGa_SEg9J4q8YAijGaCEGdvBisXv4ioR7yYsjuZLZmbxFriddYvW7vg9RzhskvaDRLy"
    PINECONE_INDEX_NAME: str = "hackrxxx"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    TOP_K_RESULTS: int = 5
    CONFIDENCE_THRESHOLD: float = 0.7
    
    # Hugging Face model for text generation (free)
    HF_MODEL_NAME: str = "microsoft/DialoGPT-medium"  # Or use "google/flan-t5-base"
    
    # Document processing settings
    MAX_TABLE_ROWS: int = 50
    MIN_CHUNK_LENGTH: int = 50