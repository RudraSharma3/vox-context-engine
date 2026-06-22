# Professional Reflection Write-up

### 1. Most Difficult Engineering Decision Made
The most critical engineering choice was implementing a localized Reciprocal Rank Fusion (RRF) pipeline merging BM25 scores with dense cosine vectors, rather than using an out-of-the-box framework tool. Balancing the distinct scoring distributions of statistical keyword matching and dense embeddings required building a customized mathematical rank-aggregation layer. This choice minimized framework bloat while drastically increasing query precision for exact code expressions and specialized corporate terminology.

### 2. Next Improvements with an Additional Week
Given an additional week, I would implement dynamic, parent-child chunk routing alongside a cross-encoder reranking stage (such as `bge-reranker-large`). Parent-child chunking indexes micro-chunks for high-precision vector matches but passes broader parent text structures to the LLM context pool, maximizing text nuance. Adding a secondary reranker stage would further optimize the precision of context sent to the generation layer.

### 3. Stack Component to Explore Deeply
I want to dive deeper into the low-level indexing mechanics of distributed vector storage clusters—specifically optimizing HNSW graphs, optimizing vector quantization techniques (like Product Quantization), and exploring the performance differences of vector search directly at the database engine level.

### 4. How AI Tools Assisted Implementation
AI development tools acted as a force multiplier for rapid structural design and tracking dependency updates. AI assisted in writing boilerplate Pydantic schema validation layers, handling the syntax changes in the modern Google GenAI library, and generating robust Docker multi-container setups. This let me focus my time on high-level system logic, data ingestion flows, and hybrid search ranking strategies.