# 🧠 Synapse - Advanced RAG Assistant (Cloud Deployment Version)

An intelligent document analysis system built with Streamlit, LangChain, HuggingFace, and Cohere that provides enhanced chat, search, summarization, and analytics capabilities with memory-aware interactions. **This is the cloud-ready deployment version optimized for Streamlit Cloud and other hosting platforms.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ragagent-clouddemo-mpjungkw6uehg7eexsaznf.streamlit.app)

## ✨ Features

**🤖 Intelligent Chat Interface**
- Context-aware conversations with your documents
- **Chat history integration** - maintains conversation context across interactions
- **Memory-enhanced responses** - references previous conversations when relevant
- Source attribution with clickable references
- Real-time streaming responses

**📊 Document Analytics**
- Database statistics and comprehensive metrics
- File type distribution analysis
- Document management and filtering tools
- Upload tracking and metadata analysis

**🔍 Enhanced Search**
- **Query processing** - LLM helps interpret and enhance search queries
- Semantic similarity search with relevance scoring
- Configurable result limits and filtering
- Search suggestions and tips

**📝 Smart Summarization**
- **Context-aware summaries** - considers conversation history for better relevance
- All documents or specific document summaries
- Structured summary output with key insights
- Downloadable summary reports

**✨ Multi-Modal Document Support**
- PDF files with intelligent text extraction
- CSV files with structured data processing
- Text files (.txt, .md, .markdown) with format preservation
- Batch file processing with metadata tracking

## 🏗️ Architecture

The system uses an **Enhanced Cloud RAG** architecture with intelligent query processing:

1. **Document Processing**: Files are loaded, chunked with optimal overlap, and embedded using Cohere API
2. **Vector Storage**: ChromaDB stores document embeddings with rich metadata (cloud-compatible)
3. **Query Enhancement**: HuggingFace LLM processes user queries to improve retrieval effectiveness
4. **Memory Integration**: Chat history and context are maintained for coherent conversations
5. **Intelligent Retrieval**: Context-aware search finds relevant document chunks
6. **Response Generation**: HuggingFace Mistral model generates responses using retrieved context and conversation memory
7. **Interface**: Streamlit provides an intuitive web interface with cyberpunk theming

## 🚀 Quick Deployment

### Option 1: Streamlit Cloud (Recommended)

1. **Fork this repository** to your GitHub account

2. **Go to [Streamlit Cloud](https://share.streamlit.io/)**

3. **Connect your GitHub** and select this repository

4. **Set environment variables** in Streamlit Cloud settings:
   ```
   HUGGINGFACE_API_TOKEN=your_huggingface_token_here
   COHERE_API_KEY=your_cohere_api_key_here
   ```

5. **Deploy automatically** - Streamlit Cloud will handle the rest!

### Option 2: Local Development

1. **Clone the repository**
```bash
git clone https://github.com/latifabs26/RAG_AGENT-Cloud_Demo.git
cd RAG_AGENT-Cloud_Demo
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env file with your API keys
```

4. **Run the application**
```bash
streamlit run main.py
```

The application will be available at `http://localhost:8501`

## 🔑 API Keys Setup

### HuggingFace API Token (Required)

1. **Visit [HuggingFace Settings](https://huggingface.co/settings/tokens)**
2. **Create a new token** with "Read" permissions
3. **Copy the token** and add it to your environment variables

**Free Tier Limits:**
- 1,000 requests per month
- Rate limited to prevent abuse
- Upgrade to Pro for higher limits

### Cohere API Key (Required)

1. **Visit [Cohere Dashboard](https://dashboard.cohere.ai/api-keys)**
2. **Create a new API key** (free tier available)
3. **Copy the key** and add it to your environment variables

**Free Tier Includes:**
- 100 API calls per month for embeddings
- Rate limited
- Upgrade for production usage

## 📁 Project Structure

```
synapse-deployment/
├── main.py                      # Main Streamlit application with UI
├── rag_utils.py                # RAG system utilities and HuggingFace integration
├── get_embedding_function.py   # Cohere embeddings configuration
├── styles.css                  # Custom cyberpunk theme (dark mode)
├── requirements.txt            # Python dependencies (cloud-ready)
├── packages.txt               # System packages for Streamlit Cloud
├── .env.example              # Environment variables template
├── .gitignore               # Git ignore rules (excludes sensitive data)
├── README.md                # Original development README
└── README-DEPLOYMENT.md    # This deployment guide
```

## ⚙️ Configuration

### Model Configuration (Cloud APIs)

**Language Model**: Mistral-7B-Instruct-v0.2 (via HuggingFace)
```python
# In rag_utils.py
model_name = "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai"
```

**Embedding Model**: Cohere embed-english-light-v3.0
```python
# In get_embedding_function.py
model = "embed-english-light-v3.0"
```

**Vector Store**: ChromaDB (cloud-compatible with persistent storage)
```python
# Automatically creates local chroma/ directory
CHROMA_PATH = "chroma"
```

### Document Processing Settings

**Text Chunking Strategy:**
```python
# For general documents
chunk_size = 800
chunk_overlap = 80

# For CSV files (larger chunks)
chunk_size = 1000
chunk_overlap = 50
```

**Supported File Extensions:**
```python
SUPPORTED_EXTENSIONS = {
    '.pdf': 'pdf',
    '.csv': 'csv', 
    '.md': 'text',
    '.markdown': 'text',
    '.txt': 'text',
}
```

### Search and Retrieval Settings

```python
# Similarity search parameters
k = 5  # Number of documents to retrieve
similarity_threshold = 0.7  # Minimum relevance score

# Chat history context
max_history_messages = 6  # Previous messages to include
```

## 🎨 User Interface

### Cyberpunk Dark Theme
- **Modern dark interface** with neon accents
- **Responsive design** - works on desktop and mobile
- **Animated elements** - smooth transitions and loading states
- **Custom styling** - Professional appearance with CSS theming

### Interface Components
- **Header**: Synapse branding with animated chat icon
- **Sidebar**: File upload, database stats, and controls
- **Main Tabs**: Chat, Search, Analytics with distinct styling
- **Chat Interface**: Message bubbles with source attribution
- **Real-time Feedback**: Loading spinners and progress indicators

## 📚 Usage Guide

### 1. Document Upload and Processing

**Step 1: Upload Documents**
- Use the sidebar file uploader
- Select multiple files (PDF, CSV, TXT, MD, Markdown)
- Supported formats are automatically detected

**Step 2: Process Files**
- Click "📥 Process Uploaded Files"
- Files are chunked and embedded using Cohere API
- Metadata is tracked (upload time, file type, source)
- Progress is shown in real-time

**Step 3: Verify Database**
- Check "Quick Stats" in sidebar
- View "Files" and "Sections" counts
- Browse uploaded sources list

### 2. Chat with Documents

**Enhanced Conversational AI:**
- Ask questions in natural language
- System maintains conversation context automatically
- **Memory-aware responses** reference previous messages
- **Intelligent query processing** enhances your questions for better results

**Example Interactions:**
```
You: "What are the main topics in the uploaded documents?"
Assistant: [Analyzes all documents and provides overview]

You: "Tell me more about the first topic"
Assistant: [Remembers previous response and expands on first topic]

You: "How does this relate to the sales data?"
Assistant: [Connects conversation context with relevant CSV data]
```

**Features:**
- Source attribution with document names and chunks
- Streaming responses for better user experience
- Export conversations for later reference
- Clear chat history when needed

### 3. Advanced Search

**Intelligent Search Processing:**
- Enter keywords, phrases, or questions
- System processes queries to improve search effectiveness
- Results show relevance scores and source information
- Configurable number of results (1-20)

**Search Tips:**
- Use specific terms for better results
- Ask questions instead of just keywords
- Reference previous conversations for context
- Combine document names with search terms

### 4. Document Summarization

**Context-Aware Summaries:**
- Summarize specific documents or entire collection
- **Memory integration** - considers your previous questions and interests
- Structured output with key insights and conclusions
- Progressive summarization improves with more context

**Summary Options:**
- Single document summaries
- Cross-document analysis
- Topic-specific summaries based on chat history
- Downloadable summary reports

### 5. Analytics and Database Management

**Database Statistics:**
- Total documents and chunks in database
- File type distribution
- Upload history and metadata
- Source document management

**Document Management:**
- View all uploaded sources
- Filter by document type or upload date
- Clear database when needed
- Track conversation patterns

## 🔧 Advanced Configuration

### Environment Variables

```env
# Required for deployment
HUGGINGFACE_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxx
COHERE_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Optional: Custom model settings
HUGGINGFACE_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
COHERE_MODEL_NAME=embed-english-light-v3.0

# Optional: Performance tuning
MAX_CHUNK_SIZE=800
CHUNK_OVERLAP=80
SEARCH_RESULTS_K=5
```

### Custom Model Integration

**Switching HuggingFace Models:**
```python
# In rag_utils.py - HuggingFaceInferenceModel class
def __init__(self, model_name="your-preferred-model", api_key=None):
    self.model_name = model_name
    # ... rest of initialization
```

**Alternative Embedding Models:**
```python
# In get_embedding_function.py
embeddings = CohereEmbeddings(
    model="embed-multilingual-v3.0",  # For multilingual support
    cohere_api_key=api_key
)
```

### Performance Optimization

**For Large Document Collections:**
```python
# Increase chunk size for faster processing
chunk_size = 1200
chunk_overlap = 120

# Reduce search results for faster responses  
k = 3
```

**For Better Memory Usage:**
```python
# Limit chat history
max_history_messages = 4

# Smaller embedding dimensions
# (Configure in Cohere dashboard)
```

## 🚨 Troubleshooting

### Common Deployment Issues

**1. SQLite Version Error (Streamlit Cloud)**
✅ **Already Fixed** - `packages.txt` includes sqlite3 and `pysqlite3-binary` is in requirements

**2. API Rate Limits**
```
Error: HuggingFace API rate limit exceeded
```
**Solutions:**
- Upgrade to HuggingFace Pro account
- Implement request caching
- Use alternative models with higher limits
- Add retry logic with backoff

**3. Environment Variables Not Found**
```
Error: HUGGINGFACE_API_TOKEN environment variable is required
```
**Solutions:**
- Verify API keys are set in Streamlit Cloud settings
- Check `.env` file format for local development
- Ensure no extra spaces in API keys
- Test API keys independently

**4. Memory Issues on Cloud Platforms**
```
Error: Memory limit exceeded during document processing
```
**Solutions:**
- Reduce chunk_size in document processing
- Process fewer files simultaneously
- Use lighter embedding models
- Implement document size limits

### Performance Issues

**Slow Response Times:**
- Check API rate limits and usage
- Reduce number of search results (k parameter)
- Optimize chunk sizes for your use case
- Consider caching frequent queries

**Large File Processing:**
- Break large files into smaller sections
- Increase processing timeouts
- Monitor memory usage during uploads
- Implement progressive file processing

### API Integration Issues

**HuggingFace API Problems:**
```python
# Test API connection
import requests
headers = {"Authorization": f"Bearer {api_token}"}
response = requests.get("https://api.huggingface.co/user", headers=headers)
print(response.status_code)  # Should be 200
```

**Cohere API Problems:**
```python
# Test embedding API
import cohere
co = cohere.Client("your-api-key")
response = co.embed(texts=["test"], model="embed-english-light-v3.0")
print(len(response.embeddings))  # Should return embeddings
```

## 🔄 Development and Customization

### Adding New Document Types

1. **Update supported extensions:**
```python
# In rag_utils.py
SUPPORTED_EXTENSIONS = {
    '.pdf': 'pdf',
    '.csv': 'csv',
    '.txt': 'text',
    '.md': 'text',
    '.docx': 'docx',  # New type
}
```

2. **Create loader function:**
```python
def load_single_docx(file_path):
    """Load a single DOCX file"""
    try:
        from docx import Document
        doc = Document(file_path)
        content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return [Document(page_content=content, metadata={"source": file_path})]
    except Exception as e:
        st.error(f"Failed to load DOCX {file_path}: {e}")
        return []
```

3. **Update processing logic:**
```python
# In process_uploaded_file function
elif file_type == 'docx':
    documents = load_single_docx(temp_file_path)
```

### Enhancing Memory Features

**Extended Conversation Context:**
```python
def format_chat_history_enhanced(messages, max_tokens=2000):
    """Enhanced chat history with token counting"""
    formatted_history = []
    token_count = 0
    
    for message in reversed(messages):
        message_text = f"{'Human' if isinstance(message, HumanMessage) else 'Assistant'}: {message.content}"
        message_tokens = len(message_text.split())  # Simple token estimation
        
        if token_count + message_tokens > max_tokens:
            break
            
        formatted_history.insert(0, message_text)
        token_count += message_tokens
    
    return "\n".join(formatted_history)
```

**Smart Context Selection:**
```python
def select_relevant_context(query, chat_history, max_messages=6):
    """Select most relevant messages for current query"""
    # Implement semantic similarity between query and previous messages
    # Return most relevant subset of chat history
    pass
```

### Custom UI Themes

**Light Theme Alternative:**
```css
/* Add to styles.css */
.light-theme {
    --primary-bg: #ffffff;
    --secondary-bg: #f8f9fa;
    --text-primary: #212529;
    --text-secondary: #6c757d;
    --accent-color: #0d6efd;
}
```

**Theme Switcher:**
```python
# Add to main.py
theme = st.sidebar.selectbox("Theme", ["Dark (Cyberpunk)", "Light"])
if theme == "Light":
    st.markdown('<link rel="stylesheet" href="light-theme.css">', unsafe_allow_html=True)
```

## 📊 Monitoring and Analytics

### Usage Tracking

**Basic Analytics:**
```python
# Track user interactions
def log_interaction(interaction_type, details):
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "type": interaction_type,
        "details": details
    }
    # Save to file or database
```

**Performance Metrics:**
```python
# Monitor response times
import time

start_time = time.time()
response = model.invoke(prompt)
response_time = time.time() - start_time

# Log metrics
st.sidebar.metric("Response Time", f"{response_time:.2f}s")
```

## 🌟 Production Deployment

### Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run application
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Docker Compose:**
```yaml
version: '3.8'
services:
  synapse:
    build: .
    ports:
      - "8501:8501"
    environment:
      - HUGGINGFACE_API_TOKEN=${HUGGINGFACE_API_TOKEN}
      - COHERE_API_KEY=${COHERE_API_KEY}
    volumes:
      - ./chroma:/app/chroma
    restart: unless-stopped
```

### Heroku Deployment

**Procfile:**
```
web: streamlit run main.py --server.port=$PORT --server.address=0.0.0.0
```

**Deploy Commands:**
```bash
# Create Heroku app
heroku create synapse-rag-app

# Set environment variables
heroku config:set HUGGINGFACE_API_TOKEN=your_token
heroku config:set COHERE_API_KEY=your_key

# Deploy
git push heroku main
```

### AWS/GCP Deployment

**Requirements for cloud deployment:**
- Environment variables for API keys
- Persistent storage for ChromaDB
- Load balancing for multiple users
- Monitoring and logging setup
- Backup strategy for document database

## 📄 License and Acknowledgments

### License
This project is open source and available under the [MIT License](LICENSE).

### Acknowledgments

**Core Technologies:**
- **HuggingFace** 🤗 - Providing free inference API for language models
- **Cohere** - High-quality embedding models with generous free tier  
- **Streamlit** - Amazing web framework for rapid deployment
- **LangChain** - Enhanced RAG framework and document processing
- **ChromaDB** - Efficient vector database with metadata support

**Models Used:**
- **Mistral-7B-Instruct-v0.2** - Language model for intelligent responses
- **embed-english-light-v3.0** - Cohere embeddings for semantic search

**Special Thanks:**
- Streamlit community for deployment best practices
- LangChain contributors for RAG improvements
- Open source contributors who made this possible

---

**🚀 Built for intelligent document processing in the cloud**

*Deploy once, analyze forever* ✨

**Created by Ben Slama Latifa**  
*Final-year Data Science Engineering Student*

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/latifabs26/RAG_AGENT-Cloud_Demo)
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red)](https://your-app-url.streamlit.app)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)