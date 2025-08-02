import requests
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, RAPIDAPI_EMBEDDING_URL, RAPIDAPI_KEY, AIPROXY_TOKEN, OPEN_AI_PROXY_URL
from pdf_utils import chunk_text
import time

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
try:
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=3072,  # Updated to match text-embedding-3-large
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)
except Exception as e:
    print(f"Warning: Could not create index due to {e}. Proceeding with existing index if available.")
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        raise Exception("No index available and creation failed. Please check your pod limit and try again.")
index = pc.Index(PINECONE_INDEX_NAME)

# Headers for API requests
rapidapi_headers = {
    "x-rapidapi-key": RAPIDAPI_KEY,
    "x-rapidapi-host": "openai-embedding-v3-large.p.rapidapi.com",
    "Content-Type": "application/json"
}
proxy_headers = {
    "Authorization": f"Bearer {AIPROXY_TOKEN}",
    "Content-Type": "application/json"
}

def get_openai_embedding(text: str) -> List[float]:
    """Get embedding from RapidAPI endpoint."""
    payload = {
        "input": text,
        "model": "text-embedding-3-large",
        "encoding_format": "float"
    }
    response = requests.post(RAPIDAPI_EMBEDDING_URL, json=payload, headers=rapidapi_headers)
    if response.status_code == 200:
        return response.json()["data"][0]["embedding"]
    else:
        raise Exception(f"Embedding API error: {response.text}")

def optimize_query(query: str) -> List[str]:
    """Optimize query using OpenAI chat completion API to detect intents."""
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that splits user queries into specific sub-queries based on intent."},
            {"role": "user", "content": f"Split the following query into specific sub-queries: {query}"}
        ],
        "max_tokens": 50
    }
    response = requests.post(OPEN_AI_PROXY_URL, json=payload, headers=proxy_headers)
    if response.status_code == 200:
        sub_queries = response.json()["choices"][0]["message"]["content"].strip().split("\n")
        return [q.strip() for q in sub_queries if q.strip()]
    else:
        print(f"Query optimization error: {response.text}")
        return [query]  # Fallback to original query

def process_and_store_chunks(chunks: List[Dict[str, Any]], document_id: str) -> None:
    """Generate embeddings and store in Pinecone using RapidAPI."""
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        if chunk["metadata"]["type"] == "table":
            sub_chunks = [text]
        else:
            sub_chunks = chunk_text(text)
        for j, sub_chunk in enumerate(sub_chunks):
            embedding = get_openai_embedding(sub_chunk)
            vector_id = f"{document_id}_chunk_{i}_{j}"
            metadata = chunk["metadata"].copy()
            metadata["chunk_index"] = j
            metadata["text"] = sub_chunk
            index.upsert([(vector_id, embedding, metadata)])
            print(f"Stored chunk {vector_id}")

def process_multi_query(query: str) -> Dict[str, Any]:
    """Process a query with multiple intents and combine results."""
    sub_queries = optimize_query(query)
    all_answers = {}
    
    for sub_query in sub_queries:
        query_embedding = get_openai_embedding(sub_query)
        results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        
        answers = []
        seen_texts = set()
        for match in results["matches"]:
            text = match["metadata"]["text"]
            if text not in seen_texts:
                answers.append(text)
                seen_texts.add(text)
                if len(answers) >= 2:
                    break
        
        all_answers[sub_query] = answers
    
    combined_answer = "\n".join(
        f"{sub_q.strip()}: {', '.join(ans for ans in all_answers[sub_q][:2])}"
        for sub_q, ans in all_answers.items()
    )
    
    top_chunk = results["matches"][0]
    return {
        "query": query,
        "answer": combined_answer if len(sub_queries) > 1 else top_chunk["metadata"]["text"],
        "source": top_chunk["metadata"],
        "confidence": top_chunk["score"],
        "sub_answers": all_answers if len(sub_queries) > 1 else {}
    }

def process_query(query: str) -> Dict[str, Any]:
    """Process a query, using multi-query logic for complex queries."""
    sub_queries = optimize_query(query)
    if len(sub_queries) > 1:
        return process_multi_query(query)
    else:
        query_embedding = get_openai_embedding(query)
        results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        top_chunk = results["matches"][0]
        return {
            "query": query,
            "answer": top_chunk["metadata"]["text"],
            "source": top_chunk["metadata"],
            "confidence": top_chunk["score"]
        }