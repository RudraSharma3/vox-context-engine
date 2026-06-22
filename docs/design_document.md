# Engineering Design Document: VoxContextEngine

## 1. Vector Database Selection: Qdrant (Containerized deployment)
For this implementation, **Qdrant** was selected as the core high-dimensional vector search engine over alternatives like PGVector, FAISS, or Pinecone.

### Alternatives Considered & Trade-offs:
* **PostgreSQL + pgvector:** Excellent for unified relational data structures but exhibits significant indexing performance degradation and slower HNSW index builds as collections scale past millions of vectors.
* **FAISS:** Exceptional execution speed for raw vector matching but lacks real-time payload filtering, dynamic CRUD vector mutations, and native multi-tenancy access controls out of the box.
* **Pinecone:** Fully managed and robust, but introduces non-deterministic cloud latency overhead, vendor lock-in, and testing bottlenecks during localized development workflows.

### Selected Choice Justification:
Qdrant was chosen because it natively combines high-performance dense vector searches with advanced sparse token matching and localized payload filtering. Running Qdrant inside a localized Docker container guarantees strict environment reproducibility, enabling seamless horizontal scaling in enterprise cloud environments without altering the client implementation layer.

## 2. Embedding Model Selection: `all-MiniLM-L6-v2`
The framework utilizes the open-source **`all-MiniLM-L6-v2`** model via the SentenceTransformers runtime library.

### Trade-offs & Expected Strengths/Weaknesses:
* **Strengths:** Blistering computational speed and minimal memory footprint. It generates a compact 384-dimensional vector space that optimizes both RAM utilization and matching latencies within our Qdrant instance. It is highly effective for technical terminology and standard sentence-level semantic mappings.
* **Weaknesses:** It features a maximum input token constraint of 256 tokens. Any structured text surpassing this sequence boundary will face truncated pooling, losing long-range document dependencies.
* **Mitigation:** This is mitigated directly via a strict character sliding-window text-chunking strategy that fits within the model's comfortable sequence boundaries.

## 3. Chunking Strategy
* **Chunk Size:** 500 characters (Approx. 70-90 tokens).
* **Overlap Strategy:** 50 characters (Approx. 10-15 tokens) sliding window threshold.

### Structural Rationale:
A smaller, dense chunking footprint ensures that every vector pushed to Qdrant contains a high concentration of semantic information without fluff. The 50-character sliding overlap acts as a bridge, preserving contextual flow and preventing broken entity names or fragmented descriptions across arbitrary document boundary splits.