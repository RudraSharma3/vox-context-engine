# src/config.py
import os

# Securely extract API keys from host environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# Part 2 System Requirements: Configurable Ingestion Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
COLLECTION_NAME = "enterprise_knowledge_base"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"