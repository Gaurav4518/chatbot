from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.environ.get('PINECONE_API_KEY')
print(f"Checking indexes for API Key: {api_key[:10]}...")

try:
    pc = Pinecone(api_key=api_key)
    indexes = pc.list_indexes()
    print(f"Found {len(indexes)} indexes:")
    for idx in indexes:
        print(f" - {idx.name} (Host: {idx.host})")
except Exception as e:
    print(f"Error listing indexes: {e}")
