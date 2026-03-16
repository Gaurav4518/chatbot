from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Optional
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os


#Extract Data From the PDF File
def load_pdf_file(data):
    """Loads all PDFs from a directory. Skips files that cannot be read."""
    if os.path.isfile(data):
        try:
            loader = PyPDFLoader(data)
            return loader.load()
        except Exception as e:
            print(f"Skipping file {data}: {e}")
            return []
    else:
        # Use silent_errors=True so one bad PDF doesn't crash the whole batch
        loader = DirectoryLoader(
            data,
            glob="*.pdf",
            loader_cls=PyPDFLoader,
            silent_errors=True  # Skip unreadable PDFs instead of crashing
        )
        return loader.load()


def load_multiple_pdfs(file_paths: List[str]) -> List[Document]:
    """Loads a specific list of PDF files. Skips any that fail."""
    all_docs = []
    for path in file_paths:
        if os.path.isfile(path):
            try:
                loader = PyPDFLoader(path)
                all_docs.extend(loader.load())
                print(f"  Loaded: {os.path.basename(path)}")
            except Exception as e:
                print(f"  Skipped: {os.path.basename(path)} — Reason: {e}")
    return all_docs


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs



#Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks



#Download the Embeddings from HuggingFace 
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions
    return embeddings


def process_and_index_pdfs(index_name: str, pinecone_api_key: str, data_path: str = None, file_list: List[str] = None):
    """
    Smarter indexing that can take either a directory (data_path) 
    OR a specific list of files (file_list).
    Skips unreadable/encrypted PDFs gracefully.
    """
    # 1. Load documents
    if file_list:
        print(f"Incremental Indexing: Loading {len(file_list)} specific files...")
        extracted_data = load_multiple_pdfs(file_list)
    elif data_path:
        print(f"Full Indexing: Loading all PDFs from {data_path}...")
        extracted_data = load_pdf_file(data=data_path)
    else:
        return "Error: No data source provided."

    if not extracted_data:
        return "No readable PDF files found to index."
    
    # 2. Filter metadata and split into smaller chunks
    filter_data = filter_to_minimal_docs(extracted_data)
    text_chunks = text_split(filter_data)
    
    # 3. Download the embedding model
    embeddings = download_hugging_face_embeddings()
    
    # 4. Connect to Pinecone and create index if it doesn't exist
    pc = Pinecone(api_key=pinecone_api_key)
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    
    # 5. Send chunks to Pinecone cloud
    PineconeVectorStore.from_documents(
        documents=text_chunks,
        index_name=index_name,
        embedding=embeddings, 
    )
    
    return f"Successfully indexed {len(text_chunks)} new chunks."