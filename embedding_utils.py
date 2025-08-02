import requests
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, AIPROXY_TOKEN, OPEN_AI_PROXY_URL, OPEN_AI_EMBEDDING_URL
from pdf_utils import chunk_text
import time

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,  # OpenAI embeddings are 1536 dimensions
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",      # or "gcp" if you prefer
            region="us-east-1"  # or another supported region
        )
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)
index = pc.Index(PINECONE_INDEX_NAME)

# Headers for API requests
headers = {
    "Authorization": f"Bearer {AIPROXY_TOKEN}",
    "Content-Type": "application/json"
}

def get_openai_embedding(text: str) -> List[float]:
    """Get embedding from OpenAI API proxy with fallback models."""
    models_to_try = ["text-embedding-3-small", "text-embedding-3-large", "text-similarity-davinci-001"]
    for model in models_to_try:
        payload = {
            "model": model,
            "input": text
        }
        response = requests.post(OPEN_AI_EMBEDDING_URL, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json()["data"][0]["embedding"]
        elif "Invalid model" not in response.text:
            raise Exception(f"Embedding API error: {response.text}")
        print(f"Model {model} not supported, trying next...")
    raise Exception("No supported embedding model found. Check proxy documentation or API support.")

def optimize_query(query: str) -> List[str]:
    """Optimize query using OpenAI chat completion API to detect intents."""
    payload = {
        "model": "gpt-3.5-turbo",  # Adjust based on available models
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that splits user queries into specific sub-queries based on intent."},
            {"role": "user", "content": f"Split the following query into specific sub-queries: {query}"}
        ],
        "max_tokens": 50
    }
    response = requests.post(OPEN_AI_PROXY_URL, json=payload, headers=headers)
    if response.status_code == 200:
        sub_queries = response.json()["choices"][0]["message"]["content"].strip().split("\n")
        return [q.strip() for q in sub_queries if q.strip()]
    else:
        print(f"Query optimization error: {response.text}")
        return [query]  # Fallback to original query

def process_and_store_chunks(chunks: List[Dict[str, Any]], document_id: str) -> None:
    """Generate embeddings and store in Pinecone using OpenAI API."""
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