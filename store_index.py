from dotenv import load_dotenv
import os
from src.helper import process_and_index_pdfs

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
DATA_PATH = os.environ.get('DATA_PATH', 'data/')
INDEX_NAME = "medical-chatbot"

if __name__ == "__main__":
    print(f"Starting Full Indexing from: {DATA_PATH}")
    
    result = process_and_index_pdfs(
        data_path=DATA_PATH,
        index_name=INDEX_NAME,
        pinecone_api_key=PINECONE_API_KEY
    )
    
    print(result)