from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.environ.get('PINECONE_API_KEY')
index_name = "medical-chatbot"

try:
    pc = Pinecone(api_key=api_key)
    print(f"Connecting to index: {index_name}")
    desc = pc.describe_index(index_name)
    print(f"Index Description: {desc}")
    
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    print(f"Index Stats: {stats}")
except Exception as e:
    print(f"Error: {e}")
