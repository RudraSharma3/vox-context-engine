# src/main.py
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types  # Explicit import to resolve NameError
import src.config as config
from src.retrieval import HybridRetriever

# 1. Define modern FastAPI lifespan instead of deprecated @app.on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup actions securely (loading indices) before accepting traffic."""
    retriever.load_bm25_store()
    yield

# Initialize FastAPI app with updated lifespan orchestration
app = FastAPI(
    title="VoxContextEngine",
    description="Production-grade, document-grounded hybrid RAG engine.",
    version="1.0.0",
    lifespan=lifespan
)

# Initialize components
retriever = HybridRetriever()

# 2. Initialize GenAI Client explicitly targeting our localized configuration key 
ai_client = genai.Client(api_key=config.GEMINI_API_KEY)

# Define Pydantic models for strict API contracts
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(payload: QueryRequest):
    """
    Processes incoming questions, executes hybrid retrieval across indexed corporate data,
    and returns an LLM response grounded strictly in the retrieved context.
    """
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        # 1. Execute Hybrid Retrieval with Reciprocal Rank Fusion
        retrieved_chunks = retriever.retrieve(payload.question, top_k=5)
        
        if not retrieved_chunks:
            return QueryResponse(
                answer="I could not find any relevant documentation to answer your question securely.",
                sources=[]
            )
            
        # 2. Extract context strings and deduplicate source file names
        context_text = "\n\n".join([f"[Source: {doc['source']}]\n{doc['content']}" for doc in retrieved_chunks])
        unique_sources = list(set([doc['source'] for doc in retrieved_chunks]))
        
        # 3. Construct System Prompt to enforce strict grounding
        system_instruction = (
            "You are an expert enterprise AI assistant. Your goal is to answer the user's question "
            "using ONLY the provided text snippets. If the context does not contain enough information "
            "to confidently answer, state that you do not have sufficient information. Do not invent or "
            "hallucinate details outside the context. Keep your response clear, concise, and business-focused."
        )
        
        # 4. Formulate user message bundle
        user_prompt = f"Context Material:\n{context_text}\n\nUser Question: {payload.question}"
        
        # 5. Call Gemini 2.5 Flash using the modern SDK explicitly bound to our verified client
        response = ai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,  # Low temperature to suppress creative hallucinations
            )
        )
        
        return QueryResponse(
            answer=response.text.strip(),
            sources=unique_sources
        )
        
    except Exception as e:
        # Log the exception on the server side and throw clean 500 error
        print(f"Error handling query: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error executing RAG execution pipeline.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)