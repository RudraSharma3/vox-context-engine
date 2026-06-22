# src/ingest.py
import os
import re
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from rank_bm25 import BM25Okapi
import pickle
import src.config as config

class IngestionPipeline:
    def __init__(self):
        self.encoder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        self.qdrant_client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        
    def extract_text(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext == '.md':
            with open(file_path, 'r', encoding='utf-8') as f:
                # Strip basic markdown headers/formatting structural text
                text = f.read()
                return re.sub(r'[#\*`\-]', '', text)
        elif ext == '.pdf':
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        return ""

    def chunk_text(self, text: str, source_name: str) -> list[dict]:
        # Simple character-based sliding window chunking
        chunks = []
        words = text.split()
        
        # Approximate chunk size in words (rough estimation: 1 token ~ 0.75 words)
        chunk_word_size = int(config.CHUNK_SIZE * 0.75)
        overlap_word_size = int(config.CHUNK_OVERLAP * 0.75)
        
        step = chunk_word_size - overlap_word_size
        if step <= 0:
            step = chunk_word_size
            
        for i in range(0, len(words), step):
            chunk_words = words[i:i + chunk_word_size]
            chunk_content = " ".join(chunk_words)
            if len(chunk_content.strip()) > 10:
                chunks.append({
                    "content": chunk_content,
                    "metadata": {"source": source_name, "chunk_id": len(chunks)}
                })
        return chunks

    def run_pipeline(self, data_dir: str):
        all_chunks = []
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"Created {data_dir} directory. Drop sample files inside it.")
            return

        for file_name in os.listdir(data_dir):
            # EXPLICIT GUARDRAIL: Skip background pickle files or data artifacts
            if file_name.endswith('.pkl'):
                continue
                
            file_path = os.path.join(data_dir, file_name)
            if os.path.isfile(file_path):
                print(f"Processing: {file_name}")
                text = self.extract_text(file_path)
                file_chunks = self.chunk_text(text, file_name)
                all_chunks.extend(file_chunks)

        if not all_chunks:
            print("No documents found to index.")
            return

        # Modern Qdrant dimension retrieval method
        vector_size = self.encoder.get_embedding_dimension()
        
        # Safe collection recreation using updated Qdrant API conventions
        if self.qdrant_client.collection_exists(config.COLLECTION_NAME):
            self.qdrant_client.delete_collection(config.COLLECTION_NAME)
            
        self.qdrant_client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

        # Generate Dense Vector Embeddings
        texts = [c["content"] for c in all_chunks]
        embeddings = self.encoder.encode(texts, show_progress_bar=True)

        # Upload to Qdrant
        payloads = [c["metadata"] for c in all_chunks]
        for idx, payload in enumerate(payloads):
            payload["content"] = texts[idx]

        self.qdrant_client.upload_collection(
            collection_name=config.COLLECTION_NAME,
            vectors=embeddings,
            payload=payloads,
            ids=list(range(len(all_chunks)))
        )

        # Fit and save local BM25 Model for Sparse Keyword Search matching
        tokenized_corpus = [text.lower().split() for text in texts]
        bm25 = BM25Okapi(tokenized_corpus)
        
        with open("data/bm25_corpus.pkl", "wb") as f:
            pickle.dump((bm25, all_chunks), f)
            
        print(f"Successfully indexed {len(all_chunks)} chunks into Qdrant and BM25 store.")

if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.run_pipeline("data")