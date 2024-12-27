from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from src.chain_builder import rag_query
import json

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class Query(BaseModel):
    query: str
    chat_history: Optional[List[Message]] = None

@app.post("/generate")
async def generate_response(query: Query):
    """Non-streaming endpoint for RAG responses"""
    # Convert chat history to list format
    history = [{"role": msg.role, "content": msg.content} 
              for msg in (query.chat_history or [])]
    
    # Get response from RAG chain (non-streaming)
    response = rag_query(query.query, history, stream=False)
    
    return {
        "answer": response['answer']
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Streaming endpoint for RAG responses"""
    await websocket.accept()
    
    try:
        while True:
            # Receive and parse the query
            data = await websocket.receive_text()
            query_data = json.loads(data)
            
            # Extract query and history
            user_query = query_data.get("query", "")
            history = query_data.get("chat_history", [])
            
            # Stream the response
            for chunk in rag_query(user_query, history, stream=True):
                await websocket.send_text(chunk)
            
            # Send end marker
            await websocket.send_text("[END]")
            
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)