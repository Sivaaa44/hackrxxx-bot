import re
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import logging
from config import Config
from answer_generator import AnswerGenerator

# Handle different Pinecone versions
try:
    from pinecone import Pinecone
    PINECONE_V3 = True
except ImportError:
    try:
        import pinecone
        PINECONE_V3 = False
    except ImportError:
        raise ImportError("Pinecone is not installed.")

logger = logging.getLogger(__name__)

class ImprovedQueryEngine:
    def __init__(self, config: Config):
        self.config = config
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        self.answer_generator = AnswerGenerator()
        
        # Initialize Pinecone
        if PINECONE_V3:
            self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
            self.index = self.pc.Index(config.PINECONE_INDEX_NAME)
        else:
            pinecone.init(api_key=config.PINECONE_API_KEY, environment="us-east-1-aws")
            self.index = pinecone.Index(config.PINECONE_INDEX_NAME)
    
    def query(self, question: str, document_id: str) -> Dict[str, Any]:
        """Process query and return answer."""
        
        # Get relevant chunks
        relevant_chunks = self._retrieve_chunks(question, document_id)
        
        # Generate answer
        chunk_texts = [chunk["metadata"]["text"] for chunk in relevant_chunks]
        answer = self.answer_generator.generate_answer(question, chunk_texts)
        
        # Calculate confidence
        confidence = self._calculate_confidence(relevant_chunks)
        
        return {
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "sources": [
                {
                    "page": chunk["metadata"]["page"],
                    "type": chunk["metadata"]["type"],
                    "score": chunk["score"]
                }
                for chunk in relevant_chunks[:2]
            ]
        }
    
    def _retrieve_chunks(self, query: str, document_id: str) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks using semantic search."""
        
        # Expand query with synonyms
        expanded_query = self._expand_query(query)
        
        # Get embedding
        query_embedding = self.embedder.encode(expanded_query).tolist()
        
        # Search in Pinecone
        results = self.index.query(
            vector=query_embedding,
            filter={"document_id": document_id},
            top_k=5,
            include_metadata=True
        )
        
        return results.get("matches", [])
    
    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms."""
        query_lower = query.lower()
        expansions = []
        
        # Add original query
        expansions.append(query)
        
        # Add domain-specific expansions
        if "grace period" in query_lower:
            expansions.extend(["premium payment", "due date", "late payment"])
        elif "waiting period" in query_lower:
            expansions.extend(["pre-existing", "PED", "months coverage"])
        elif "maternity" in query_lower:
            expansions.extend(["pregnancy", "childbirth", "delivery"])
        elif "room rent" in query_lower:
            expansions.extend(["ICU", "hospital charges", "room limit"])
        
        return " ".join(expansions)
    
    def _calculate_confidence(self, chunks: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on retrieval results."""
        if not chunks:
            return 0.0
        
        # Average of top scores
        scores = [chunk["score"] for chunk in chunks[:3]]
        avg_score = sum(scores) / len(scores)
        
        # Convert to percentage
        confidence = min(avg_score * 100, 95.0)  # Cap at 95%
        return round(confidence, 1)