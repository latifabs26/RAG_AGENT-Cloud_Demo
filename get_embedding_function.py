from langchain_cohere import CohereEmbeddings
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_embedding_function():
    """Get Cohere embeddings function for LangChain"""
    # Get API key from environment
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError("COHERE_API_KEY environment variable is required")
    
    embeddings = CohereEmbeddings(
        model="embed-english-light-v3.0",
        cohere_api_key=api_key
    )
    return embeddings

#we can use LangSmith later to evaluate teh performance