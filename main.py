# main.py - FastAPI Application
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
from document_processor import DocumentProcessor
from query_engine import ImprovedQueryEngine
from config import Config
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced Insurance Document Query System", version="2.0.0")

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class DetailedQueryResponse(BaseModel):
    answers: List[Dict[str, Any]]

# Initialize components
config = Config()
doc_processor = DocumentProcessor(config)
query_engine = ImprovedQueryEngine(config)

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_queries(request: QueryRequest):
    """Process documents and answer questions with enhanced accuracy."""
    try:
        logger.info(f"Processing document: {request.documents}")
        
        # Process document and create embeddings
        document_id = doc_processor.process_document(request.documents)
        
        # Process all questions
        answers = []
        for i, question in enumerate(request.questions):
            logger.info(f"Processing question {i+1}/{len(request.questions)}: {question}")
            result = query_engine.query(question, document_id)
            answers.append(result["answer"])
            logger.info(f"Answer confidence: {result['confidence']}%")
        
        return QueryResponse(answers=answers)
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
if __name__ == "__main__":
    print("🚀 Starting Enhanced Insurance Document Query System...")
    uvicorn.run(app, host="0.0.0.0", port=8001)