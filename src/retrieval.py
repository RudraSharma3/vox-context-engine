# src/retrieval.py
import pickle
import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import src.config as config

class HybridRetriever:
    def __init__(self):
        self.encoder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        self.qdrant_client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        self.bm25 = None
        self.corpus_chunks = None
        self.load_bm25_store()

    def load_bm25_store(self) -> None:
        """Loads the localized BM25 index generated during ingestion."""
        bm25_path = "data/bm25_corpus.pkl"
        if os.path.exists(bm25_path):
            with open(bm25_path, "rb") as f:
                self.bm25, self.corpus_chunks = pickle.load(f)
        else:
            print("Warning: BM25 store not found. Run ingestion first.")

    def dense_search(self, query: str, top_k: int = 10) -> list[dict]:
        """Retrieves semantically similar chunks from Qdrant using the query_points API."""
        query_vector = self.encoder.encode(query).tolist()
        
        search_results = self.qdrant_client.query_points(
            collection_name=config.COLLECTION_NAME,
            query=query_vector,
            limit=top_k
        )
        
        results = []
        for res in search_results.points:
            results.append({
                "content": res.payload["content"],
                "source": res.payload["source"],
                "score": res.score
            })
        return results

    def sparse_search(self, query: str, top_k: int = 10) -> list[dict]:
        """Retrieves exact keyword matches using BM25."""
        if not self.bm25 or not self.corpus_chunks:
            return []
            
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include active keyword matches
                chunk = self.corpus_chunks[idx]
                results.append({
                    "content": chunk["content"],
                    "source": chunk["metadata"]["source"],
                    "score": scores[idx]
                })
        return results

    def reciprocal_rank_fusion(self, dense_results: list[dict], sparse_results: list[dict], k: int = 60, top_n: int = 5) -> list[dict]:
        """Merges ranked vector lists using Reciprocal Rank Fusion (RRF)."""
        rrf_scores = {}
        content_map = {}
        
        def calculate_rrf(results_list):
            for rank, doc in enumerate(results_list):
                doc_id = doc["content"]  # Use content string as a unique identifier
                content_map[doc_id] = doc["source"]
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0.0
                rrf_scores[doc_id] += 1.0 / (k + (rank + 1))

        calculate_rrf(dense_results)
        calculate_rrf(sparse_results)
        
        # Sort documents based on aggregated RRF value
        sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:top_n]
        
        final_retrieved = []
        for content, score in sorted_docs:
            final_retrieved.append({
                "content": content,
                "source": content_map[content]
            })
        return final_retrieved

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """Orchestrates the entire hybrid retrieval workflow."""
        dense_res = self.dense_search(query, top_k=top_k)
        sparse_res = self.sparse_search(query, top_k=top_k)
        
        # Safe fallback: if sparse search is empty, boost dense candidates pool size
        if not sparse_res:
            return dense_res[:5]
            
        return self.reciprocal_rank_fusion(dense_res, sparse_res, top_n=5)