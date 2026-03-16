from dotenv import load_dotenv
import os

load_dotenv()
key = os.environ.get('PINECONE_API_KEY')
print(f"Key length: {len(key) if key else 0}")
if key:
    print("Starts with quote: " + str(key.startswith('"')))
    print("Ends with quote: " + str(key.endswith('"')))
    print(f"First 10 chars: {key[:10]}")
