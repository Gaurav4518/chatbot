import os
from dotenv import load_dotenv
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
os.environ["PINECONE_API_KEY"] = os.environ.get('PINECONE_API_KEY')

embeddings = download_hugging_face_embeddings()
docsearch = PineconeVectorStore.from_existing_index(
    index_name="medical-chatbot",
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":8})
docs = retriever.invoke("what typhoid?")

for i, doc in enumerate(docs):
    print(f"\n--- Document {i+1} ---")
    print(doc.page_content)
