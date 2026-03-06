# MedAI: Premium RAG-powered Medical Chatbot

MedAI is a professional-grade medical assistant built using a Retrieval-Augmented Generation (RAG) pipeline. It leverages LangChain, Pinecone, and Groq (Llama 3) to provide accurate, context-aware medical information based on a provided medical encyclopedia.

---

## 🚀 Features
- **Conversational Memory**: Remembers past interactions for contextual follow-ups.
- **High-Performance RAG**: Uses Pinecone vector database for fast and relevant medical document retrieval.
- **Premium UI**: Modern dark-mode interface with Glassmorphism, smooth animations, and a custom 3D mascot.
- **Safety First**: Implemented strict medical boundaries to ensure users are prompted to consult professionals.

---

## 💻 System Requirements

### Hardware
- **Processor**: Dual-core 2.0GHz or higher.
- **RAM**: 4GB Minimum (8GB Recommended).
- **Disk Space**: ~1GB for environment and dependencies.

### Software
- **OS**: Windows, macOS, or Linux.
- **Python**: 3.10 or 3.11.
- **Git**: Installed for version control.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Gaurav4518/chatbot.git
cd chatbot
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
```

### 3. Install Dependencies
```bash
.\venv\Scripts\pip install langchain==0.3.15 langchain-community==0.3.15 langchain-core==0.3.31 langchain-groq==0.2.2 langchain-pinecone==0.2.1 langchain-text-splitters==0.3.4 langchain-openai==0.2.13 flask pypdf python-dotenv sentence-transformers
```

### 4. Environment Variables
Create a `.env` file in the root directory and add your keys:
```env
PINECONE_API_KEY = "your_pinecone_api_key"
GROQ_API_KEY = "your_groq_api_key"
```

### 5. Indexing the Medical Data
Ensure your medical PDF is in the `data/` folder, then run:
```bash
.\venv\Scripts\python store_index.py
```

---

## 🏃 Running the Application
To start the chatbot, run:
```bash
.\venv\Scripts\python app.py
```
Open your browser and navigate to: `http://localhost:8080`

---

## 📂 Project Structure
- `app.py`: Flask application core logic.
- `src/`: Custom helper functions and prompts.
- `static/`: CSS, JS, and UI assets (Avatar).
- `templates/`: HTML interface.
- `research/`: Jupyter notebooks for testing and experimentation.
- `data/`: Source medical documents.

---

## ⚠️ Disclaimer
*This chatbot is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.*
