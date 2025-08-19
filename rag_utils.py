import streamlit as st
import sys
# Fix sqlite3 version for Streamlit Cloud
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import io
import os
import tempfile
import shutil
from pathlib import Path
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from openai import OpenAI  # Changed from ollama to openai for HF inference
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.documents import Document
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from get_embedding_function import get_embedding_function
import time
import pandas as pd
from collections import Counter
import re
from datetime import datetime
from dotenv import load_dotenv  # Added for .env file support

# Load environment variables from .env file
load_dotenv()

# Set UTF-8 encoding for the environment
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Constants
CHROMA_PATH = "chroma"
DATA_PATH = "data"

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    '.pdf': 'pdf',
    '.csv': 'csv',
    '.md': 'text',
    '.markdown': 'text',
    '.txt': 'text',
}

# Enhanced prompt templates
CHAT_PROMPT_TEMPLATE = """
You are a helpful assistant that answers questions based on the provided context and chat history.

Chat History:
{chat_history}

Context from documents:
{context}

---

Current Question: {question}

Please provide a detailed answer based on the context above. If the chat history is relevant to the current question, consider it in your response. If you cannot find the answer in the provided context, say "I don't have enough information in the provided documents to answer this question."
"""

SUMMARY_PROMPT_TEMPLATE = """
You are an expert summarizer. Based on the following documents, create a comprehensive summary.

Documents:
{context}

Please provide a well-structured summary that covers:
1. Main topics and themes
2. Key findings or insights
3. Important details and statistics
4. Overall conclusions

Summary:
"""

class HuggingFaceInferenceModel:
    """Custom wrapper for Hugging Face Inference API to match Ollama interface"""
    
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2:featherless-ai", api_key=None):
        self.model_name = model_name
        # Try multiple environment variable names for API key
        if not api_key:
            api_key = (
                os.getenv("HUGGINGFACE_API_TOKEN") or 
                os.getenv("HF_TOKEN") or 
                os.getenv("HUGGINGFACE_API_KEY")
            )
        
        if not api_key:
            raise ValueError("Hugging Face API key not found. Please set HUGGINGFACE_API_TOKEN in your .env file")
        
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key
        )
    
    def invoke(self, prompt, max_tokens=2000, temperature=0.7):
        """Invoke the model with a prompt (mimics Ollama's invoke method)"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return completion.choices[0].message.content
        except Exception as e:
            st.error(f"Error calling Hugging Face API: {str(e)}")
            return "Sorry, I encountered an error while generating the response."

def load_css():
    """Load CSS from external file"""
    css_file = Path("styles.css")
    if css_file.exists():
        with open(css_file, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Fallback to basic professional styling
        st.markdown("""
        <style>
        .main { 
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            color: #1e293b;
        }
        .custom-header { 
            background: linear-gradient(135deg, #2563eb 0%, #06b6d4 100%);
            padding: 3rem 2rem; 
            border-radius: 1rem; 
            text-align: center; 
            margin-bottom: 2rem;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        }
        .custom-header h1 { 
            color: white; 
            font-size: 2.5rem; 
            font-weight: 700;
            margin: 0; 
            letter-spacing: -0.025em;
        }
        .custom-header p { 
            color: rgba(255,255,255,0.95); 
            font-size: 1.375rem; 
            font-weight: 500;
            margin: 0.5rem 0 0 0;
            text-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border: 1px solid #e2e8f0;
            text-align: center;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #2563eb;
        }
        .metric-label {
            color: #64748b;
            font-size: 0.875rem;
        }
        .summary-card {
            background: #f8fafc;
            border-left: 4px solid #2563eb;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .summary-title {
            font-weight: bold;
            color: #1e293b;
            margin-bottom: 1rem;
        }
        .summary-content {
            white-space: pre-wrap;
            line-height: 1.6;
        }
        .source-item {
            background: #f1f5f9;
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
            margin: 0.25rem 0;
            border-left: 3px solid #3b82f6;
        }
        .search-result {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 0.5rem 0;
            position: relative;
        }
        .search-result-score {
            position: absolute;
            top: 0.5rem;
            right: 0.5rem;
            background: #3b82f6;
            color: white;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.75rem;
        }
        .search-result-title {
            font-weight: bold;
            color: #1e293b;
            margin-bottom: 0.5rem;
        }
        .search-result-content {
            color: #64748b;
            line-height: 1.5;
        }
        .filter-item {
            background: #f8fafc;
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
            margin: 0.25rem 0;
        }
        .file-info {
            background: #f1f5f9;
            padding: 0.5rem;
            border-radius: 0.25rem;
            margin: 0.25rem 0;
            font-size: 0.875rem;
        }
        .user-message {
            background: #e0f2fe;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            border-left: 4px solid #0284c7;
        }
        .assistant-message {
            background: #f0f9ff;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            border-left: 4px solid #2563eb;
        }
        .fade-in {
            animation: fadeIn 0.5s ease-in;
        }
        .slide-in {
            animation: slideIn 0.3s ease-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes slideIn {
            from { transform: translateY(20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        </style>
        """, unsafe_allow_html=True)

def initialize_rag_system():
    """Initialize the RAG system components"""
    try:
        embedding_function = get_embedding_function()
        
        # Ensure chroma directory exists
        os.makedirs(CHROMA_PATH, exist_ok=True)
        
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        # Initialize HuggingFace Inference model instead of Ollama
        try:
            model = HuggingFaceInferenceModel()
        except ValueError as e:
            st.error(f"❌ {str(e)}")
            st.info("💡 Make sure your .env file contains: HUGGINGFACE_API_TOKEN=your_token_here")
            st.info("🔗 Get a free API key from: https://huggingface.co/settings/tokens")
            st.stop()
        except Exception as e:
            st.error(f"❌ Failed to initialize Hugging Face model: {str(e)}")
            st.stop()
        
        return db, model
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        raise e

def load_single_pdf(file_path):
    """Load a single PDF file"""
    try:
        loader = PyPDFLoader(file_path)
        return loader.load()
    except Exception as e:
        st.error(f"Failed to load PDF {Path(file_path).name}: {e}")
        return []

def load_single_csv(file_path):
    """Load a single CSV file with error handling"""
    try:
        loader = CSVLoader(file_path=file_path)
        return loader.load()
    except Exception as e1:
        try:
            loader = CSVLoader(
                file_path=file_path,
                encoding='utf-8',
                csv_args={'delimiter': ','}
            )
            return loader.load()
        except Exception as e2:
            try:
                loader = CSVLoader(
                    file_path=file_path,
                    encoding='utf-8',
                    csv_args={'delimiter': ';'}
                )
                return loader.load()
            except Exception as e3:
                st.error(f"Failed to load CSV {Path(file_path).name}: Multiple encoding/delimiter attempts failed")
                return []

def load_single_text(file_path):
    """Load a single text file with encoding handling"""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            loader = TextLoader(file_path, encoding=encoding)
            return loader.load()
        except Exception as e:
            continue
    
    st.error(f"Failed to load text file {Path(file_path).name} with any encoding")
    return []

def process_uploaded_file(uploaded_file, temp_dir):
    """Process a single uploaded file and return documents"""
    file_extension = Path(uploaded_file.name).suffix.lower()
    
    if file_extension not in SUPPORTED_EXTENSIONS:
        st.warning(f"Unsupported file type: {uploaded_file.name}")
        return []
    
    # Save uploaded file to temp directory
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load documents based on file type
    file_type = SUPPORTED_EXTENSIONS[file_extension]
    
    if file_type == 'pdf':
        documents = load_single_pdf(temp_file_path)
    elif file_type == 'csv':
        documents = load_single_csv(temp_file_path)
    elif file_type == 'text':
        documents = load_single_text(temp_file_path)
    else:
        return []
    
    # Update metadata to include original filename
    for doc in documents:
        doc.metadata["source"] = uploaded_file.name
        doc.metadata["uploaded"] = True
        doc.metadata["upload_time"] = datetime.now().isoformat()
    
    return documents

def split_documents(documents: list[Document]):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    
    csv_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    
    all_chunks = []
    
    for doc in documents:
        source = doc.metadata.get("source", "")
        
        try:
            if source.lower().endswith('.csv'):
                chunks = csv_splitter.split_documents([doc])
            else:
                chunks = text_splitter.split_documents([doc])
            
            all_chunks.extend(chunks)
        except Exception as e:
            st.error(f"Error splitting document {source}: {e}")
    
    return all_chunks

def calculate_chunk_ids(chunks):
    """Calculate unique IDs for chunks"""
    last_source_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        
        if source.lower().endswith('.pdf'):
            page = chunk.metadata.get("page", 0)
            current_source_id = f"{source}:{page}"
        elif source.lower().endswith('.csv'):
            row = chunk.metadata.get("row", 0)
            current_source_id = f"{source}:{row}"
        else:
            current_source_id = f"{source}:0"

        if current_source_id == last_source_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_source_id}:{current_chunk_index}"
        last_source_id = current_source_id
        chunk.metadata["id"] = chunk_id

    return chunks

def add_documents_to_chroma(chunks: list[Document], db):
    """Add new documents to the Chroma database"""
    if not chunks:
        return 0
    
    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        return len(new_chunks)
    else:
        return 0

def format_chat_history(messages):
    """Format chat history for the prompt"""
    if not messages:
        return "No previous conversation."
    
    formatted_history = []
    for message in messages[-6:]:
        if isinstance(message, HumanMessage):
            formatted_history.append(f"Human: {message.content}")
        elif isinstance(message, AIMessage):
            formatted_history.append(f"Assistant: {message.content}")
    
    return "\n".join(formatted_history)

def query_rag_with_history(query_text: str, chat_history, db, model):
    """Query RAG system with chat history context"""
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    history_text = format_chat_history(chat_history)
    
    prompt_template = ChatPromptTemplate.from_template(CHAT_PROMPT_TEMPLATE)
    prompt = prompt_template.format(
        context=context_text, 
        question=query_text,
        chat_history=history_text
    )
    
    response_text = model.invoke(prompt)
    sources = [doc.metadata.get("id", "Unknown") for doc, _score in results]
    
    return response_text, sources

def generate_summary(db, model, doc_filter=None):
    """Generate a summary of documents in the database"""
    try:
        # Get all documents or filtered documents
        if doc_filter:
            # Filter documents by source
            all_items = db.get(include=["documents", "metadatas"])
            filtered_docs = []
            for i, metadata in enumerate(all_items["metadatas"]):
                if metadata and metadata.get("source") == doc_filter:
                    filtered_docs.append(all_items["documents"][i])
            context_text = "\n\n---\n\n".join(filtered_docs[:10])  # Limit to 10 chunks
        else:
            # Get top documents by similarity to "summary overview main points"
            results = db.similarity_search("summary overview main points", k=10)
            context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
        
        if not context_text.strip():
            return "No documents available for summarization."
        
        prompt_template = ChatPromptTemplate.from_template(SUMMARY_PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text)
        
        summary = model.invoke(prompt)
        return summary
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def search_documents(db, query, k=10):
    """Search documents and return results with scores"""
    try:
        results = db.similarity_search_with_score(query, k=k)
        return results
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

def get_database_stats(db):
    """Get comprehensive statistics about the database"""
    try:
        existing_items = db.get(include=["metadatas"])
        total_docs = len(existing_items["ids"])
        
        sources = set()
        file_types = Counter()
        upload_dates = []
        
        for metadata in existing_items["metadatas"]:
            if metadata and "source" in metadata:
                sources.add(metadata["source"])
                file_ext = Path(metadata["source"]).suffix.lower()
                file_types[file_ext] += 1
                
                if "upload_time" in metadata:
                    upload_dates.append(metadata["upload_time"])
        
        return {
            "total_chunks": total_docs,
            "unique_sources": len(sources),
            "sources": list(sources),
            "file_types": dict(file_types),
            "upload_dates": upload_dates
        }
    except:
        return {
            "total_chunks": 0,
            "unique_sources": 0,
            "sources": [],
            "file_types": {},
            "upload_dates": []
        }

def create_custom_header():
    """Create a sophisticated custom header"""
    st.markdown("""
    <div class="custom-header fade-in">
        <h1>SYNAPSE</h1>
        <p>Your Advanced RAG Assistant</p>
    </div>
    """, unsafe_allow_html=True)

def create_metric_card(title, value, icon_class="metric"):
    """Create a sophisticated metric card"""
    st.markdown(f"""
    <div class="metric-card slide-in">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{title}</div>
    </div>
    """, unsafe_allow_html=True)

def display_sources_beautifully(sources):
    """Display sources in a clean format"""
    if sources:
        st.markdown("**Sources:**")
        for i, source in enumerate(sources, 1):
            st.markdown(f"""
            <div class="source-item slide-in">
                <strong>{i}.</strong> {source}
            </div>
            """, unsafe_allow_html=True)

def display_search_results(results):
    """Display search results in a clean format"""
    if not results:
        st.info("No results found.")
        return
    
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source", "Unknown")
        content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
        
        st.markdown(f"""
        <div class="search-result slide-in">
            <div class="search-result-score">{score:.3f}</div>
            <div class="search-result-title">{source}</div>
            <div class="search-result-content">{content_preview}</div>
        </div>
        """, unsafe_allow_html=True)

# Streamlit App Configuration
def main():
    st.set_page_config(
        page_title="Advanced RAG Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load CSS
    load_css()
    
    # Create header
    create_custom_header()
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "db" not in st.session_state:
        st.session_state.db = None
    if "model" not in st.session_state:
        st.session_state.model = None
    
    # Initialize RAG system
    if st.session_state.db is None or st.session_state.model is None:
        with st.spinner("🚀 Initializing RAG system..."):
            try:
                db, model = initialize_rag_system()
                st.session_state.db = db
                st.session_state.model = model
            except Exception as e:
                st.error(f"❌ Failed to initialize: {str(e)}")
                st.stop()
    
    # Sidebar for configuration and file upload
    with st.sidebar:
        st.markdown("### 📁 Document Management")
        
        # File upload section
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'csv', 'txt', 'md'],
            accept_multiple_files=True,
            help="Upload PDF, CSV, TXT, or Markdown files"
        )
        
        if uploaded_files:
            if st.button("📥 Process Uploaded Files", type="primary"):
                with st.spinner("Processing uploaded files..."):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        all_documents = []
                        
                        for uploaded_file in uploaded_files:
                            documents = process_uploaded_file(uploaded_file, temp_dir)
                            all_documents.extend(documents)
                            
                            if documents:
                                st.success(f"✅ Loaded {uploaded_file.name}")
                            else:
                                st.error(f"❌ Failed to load {uploaded_file.name}")
                        
                        if all_documents:
                            chunks = split_documents(all_documents)
                            new_chunks_count = add_documents_to_chroma(chunks, st.session_state.db)
                            
                            if new_chunks_count > 0:
                                st.success(f"✅ Added {new_chunks_count} new chunks to database")
                            else:
                                st.info("ℹ️ All documents already exist in database")
        
        # Database statistics
        st.markdown("### 📊 Database Statistics")
        stats = get_database_stats(st.session_state.db)
        
        col1, col2 = st.columns(2)
        with col1:
            create_metric_card("Total Chunks", stats["total_chunks"])
        with col2:
            create_metric_card("Unique Sources", stats["unique_sources"])
        
        if stats["sources"]:
            st.markdown("**📄 Sources:**")
            for source in stats["sources"]:
                st.markdown(f"""
                <div class="file-info">
                    {source}
                </div>
                """, unsafe_allow_html=True)
        
        # Clear database option
        if st.button("🗑️ Clear Database", type="secondary"):
            if st.session_state.db:
                try:
                    # Clear all documents
                    existing_items = st.session_state.db.get(include=[])
                    if existing_items["ids"]:
                        st.session_state.db.delete(ids=existing_items["ids"])
                        st.success("✅ Database cleared successfully!")
                    else:
                        st.info("ℹ️ Database is already empty")
                except Exception as e:
                    st.error(f"❌ Error clearing database: {str(e)}")
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🔍 Search", "📋 Summary"])
    
    with tab1:
        st.markdown("### Chat with your documents")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                st.markdown(f"""
                <div class="user-message">
                    <strong>You:</strong> {message.content}
                </div>
                """, unsafe_allow_html=True)
            elif isinstance(message, AIMessage):
                st.markdown(f"""
                <div class="assistant-message">
                    <strong>Assistant:</strong> {message.content}
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input
        if user_input := st.chat_input("Ask a question about your documents..."):
            # Add user message to history
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            
            # Get response from RAG system
            with st.spinner("🤔 Thinking..."):
                response, sources = query_rag_with_history(
                    user_input, 
                    st.session_state.chat_history[:-1],  # Exclude current message
                    st.session_state.db, 
                    st.session_state.model
                )
            
            # Add assistant response to history
            st.session_state.chat_history.append(AIMessage(content=response))
            
            # Display the new response
            st.markdown(f"""
            <div class="user-message">
                <strong>You:</strong> {user_input}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="assistant-message">
                <strong>Assistant:</strong> {response}
            </div>
            """, unsafe_allow_html=True)
            
            display_sources_beautifully(sources)
            
            # Auto-scroll to bottom
            st.rerun()
    
    with tab2:
        st.markdown("### 🔍 Search Documents")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input("Enter search query:", placeholder="e.g., machine learning algorithms")
        with col2:
            search_k = st.number_input("Results", min_value=1, max_value=20, value=5)
        
        if st.button("🔍 Search", type="primary") and search_query:
            with st.spinner("Searching documents..."):
                results = search_documents(st.session_state.db, search_query, k=search_k)
                
                if results:
                    st.markdown(f"### Found {len(results)} results:")
                    display_search_results(results)
                else:
                    st.info("No results found. Try different keywords.")
    
    with tab3:
        st.markdown("### 📋 Document Summary")
        
        # Summary options
        stats = get_database_stats(st.session_state.db)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if stats["sources"]:
                summary_filter = st.selectbox(
                    "Select Document to Summarize:",
                    stats["sources"],
                    help="Choose a specific document to summarize"
                )
            else:
                st.info("No documents available for summarization.")
                return
        
        if st.button("📋 Generate Summary", type="primary"):
            if stats["total_chunks"] == 0:
                st.warning("⚠️ No documents in database. Please upload some files first.")
            else:
                with st.spinner("Generating summary..."):
                    doc_filter = summary_filter
                    
                    summary = generate_summary(st.session_state.db, st.session_state.model, doc_filter)
                    
                    st.markdown(f"""
                    <div class="summary-card">
                        <div class="summary-title">📋 Document Summary</div>
                        <div class="summary-content">{summary}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Clear chat history option
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.success("✅ Chat history cleared!")
            st.rerun()

if __name__ == "__main__":
    main()