from flask import Flask, render_template, jsonify, request, session
from src.helper import download_hugging_face_embeddings, process_and_index_pdfs
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from src.prompt import *
import os
from werkzeug.utils import secure_filename
import uuid


app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for sessions

# Configuration for File Uploads
UPLOAD_FOLDER = 'data/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY=os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot" 

# Global variables for the RAG chain components
rag_chain = None

def init_rag_chain():
    """Initializes the RAG chain. Returns True if successful, False otherwise."""
    global rag_chain
    try:
        # 1. Connect to Pinecone
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        
        # 2. Setup Retriever
        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":8})
        
        # 3. Setup LLM
        chat_model = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY)
        
        # 4. Create contextualize retriever (Chat History Aware)
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            chat_model, retriever, contextualize_q_prompt
        )
        
        # 5. Create QA chain
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(chat_model, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        print("SUCCESS: Connected to Pinecone index successfully!")
        return True

    except Exception as e:
        print(f"ERROR: Pinecone initialization failed: {e}")
        return False

# Initial attempt at startup
init_rag_chain()

# Store history in a dictionary keyed by session ID to avoid global collision
# In production, use Redis or a database.
session_histories = {}


@app.route("/")
def index():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template('chat.html')


@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        
        result = process_and_index_pdfs(
            file_list=[path], 
            index_name=index_name, 
            pinecone_api_key=PINECONE_API_KEY
        )
        # Re-initialize chain after new data is indexed
        init_rag_chain()
        return result
    
    return "Invalid file type. Please upload a PDF."


@app.route("/upload_folder", methods=["POST"])
def upload_folder():
    if 'files[]' not in request.files:
        return "No files part"
    
    files = request.files.getlist('files[]')
    uploaded_file_paths = []
    
    for file in files:
        if file and file.filename.endswith('.pdf'):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            uploaded_file_paths.append(path)
    
    if len(uploaded_file_paths) > 0:
        result = process_and_index_pdfs(
            file_list=uploaded_file_paths, 
            index_name=index_name, 
            pinecone_api_key=PINECONE_API_KEY
        )
        # Re-initialize chain after new data is indexed
        init_rag_chain()
        return f"Uploaded {len(uploaded_file_paths)} PDFs. {result}"
    
    return "No valid PDF files found in the folder."


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    
    # Ensure chain is initialized (retry if it was missing initially)
    if rag_chain is None:
        if not init_rag_chain():
            return "ERROR: Connection to Pinecone failed. Please ensure the index exists and is populated."
    
    # Get or create history for this session
    session_id = session.get("session_id", str(uuid.uuid4()))
    if session_id not in session_histories:
        session_histories[session_id] = []
    
    chat_history = session_histories[session_id]
    
    try:
        response = rag_chain.invoke({"input": msg, "chat_history": chat_history})
        answer = response["answer"]
        
        # Save messages to history
        chat_history.extend([
            HumanMessage(content=msg),
            AIMessage(content=answer),
        ])
        
        # Keep history manageable (last 10 messages)
        if len(chat_history) > 10:
            session_histories[session_id] = chat_history[-10:]
            
        print("Response : ", answer)
        return str(answer)
        
    except Exception as e:
        print(f"Error during chat: {e}")
        return f"Sorry, an error occurred: {str(e)}"


if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
