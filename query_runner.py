from pdf_utils import fetch_data, extract_pdf_content, save_extracted_data
from embedding_utils import process_and_store_chunks, process_query

PDF_URL = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"
PDF_PATH = "policy.pdf"
DOCUMENT_ID = "arogya_sanjeevani_policy"

if __name__ == "__main__":
    ##Uncomment and run these lines only once to set up embeddings with OpenAI API
    fetch_data(PDF_URL, PDF_PATH)
    pages_data = extract_pdf_content(PDF_PATH)
    all_chunks = [chunk for page in pages_data for chunk in page["chunks"]]
    process_and_store_chunks(all_chunks, DOCUMENT_ID)
    save_extracted_data(pages_data)

    #Change your queries here without re-embedding
    sample_queries = [
        "What is the waiting period for cataract surgery and the benefits of cataract treatment?",
        "What is the premium cost?"
    ]
    for query in sample_queries:
        result = process_query(query)
        print("Sample query result:", result)