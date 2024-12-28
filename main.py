from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from chain_builder import rag_query

app = FastAPI(title="Medical Guidelines RAG API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify this in production to your specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # Extract the last user message and chat history
    user_message = request.messages[-1].content
    chat_history = request.messages[:-1]
    
    # Get response from RAG
    response = rag_query(user_message, [c.model_dump() for c in chat_history])
    return {
        "role": "assistant",
        "content": response
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)