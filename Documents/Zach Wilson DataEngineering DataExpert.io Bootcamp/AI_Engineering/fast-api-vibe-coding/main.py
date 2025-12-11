import os
import asyncio
import mmh3
import json
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from pymilvus import MilvusClient
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MILVUS_URI = os.getenv("MILVUS_URI", "https://in03-4efcec782ae2f4c.serverless.gcp-us-west1.cloud.zilliz.com")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")  # Get from .env file - no default for security
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIM = 3072 if "3-large" in EMBEDDING_MODEL else 1536

# Initialize OpenAI client
if OPENAI_API_KEY:
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
else:
    client = None

# Pydantic models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    conversation_history: List[ChatMessage] = Field(default=[], description="Conversation history")

class ChatResponse(BaseModel):
    response: str = Field(..., description="AI response")
    sources: List[Dict[str, Any]] = Field(default=[], description="RAG sources")

class AddDocumentRequest(BaseModel):
    text: str = Field(..., description="Document text to add")
    metadata: str = Field(default="{}", description="Optional metadata for the document (JSON string)")

# Milvus client
milvus_client = None

def connect_to_milvus():
    """Connect to Milvus database using MilvusClient."""
    global milvus_client
    try:
        if not MILVUS_TOKEN:
            print("Warning: MILVUS_TOKEN not set in environment variables")
            return False
        
        if not MILVUS_URI:
            print("Warning: MILVUS_URI not set in environment variables")
            return False
        
        print(f"Attempting to connect to Milvus at: {MILVUS_URI}")
        milvus_client = MilvusClient(
            uri=MILVUS_URI,
            token=MILVUS_TOKEN
        )
        
        # Test the connection by listing collections
        try:
            collections = milvus_client.list_collections()
            print(f"Connected to Milvus successfully. Found {len(collections)} collections.")
            return True
        except Exception as test_error:
            print(f"Connection test failed: {test_error}")
            return False
            
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        import traceback
        traceback.print_exc()
        return False

def setup_milvus_collection():
    """Setup Milvus collection for storing embeddings."""
    global milvus_client
    if not milvus_client:
        return
        
    try:
        # Check if collection exists
        collections = milvus_client.list_collections()
        if COLLECTION_NAME in collections:
            print(f"Collection {COLLECTION_NAME} already exists")
            return
        
        # Create collection with schema
        schema = {
            "fields": [
                {"name": "id", "dtype": "INT64", "is_primary": True},
                {"name": "text", "dtype": "VARCHAR", "max_length": 65535},
                {"name": "embedding", "dtype": "FLOAT_VECTOR", "dim": EMBEDDING_DIM},
                {"name": "channel_name", "dtype": "VARCHAR", "max_length": 128},
                {"name": "metadata", "dtype": "VARCHAR", "max_length": 65535},
            ],
            "description": "Chat embeddings collection"
        }
        
        milvus_client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema
        )
        
        # Create index
        milvus_client.create_index(
            collection_name=COLLECTION_NAME,
            field_name="embedding",
            index_params={
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
        )
        
        print(f"Collection {COLLECTION_NAME} created successfully")
        
    except Exception as e:
        print(f"Failed to setup Milvus collection: {e}")

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up...")
    if connect_to_milvus():
        setup_milvus_collection()
    yield
    # Shutdown
    print("Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="ChatGPT RAG API",
    version="1.0.0",
    lifespan=lifespan
)

# Utility functions
async def get_embedding(text: str) -> List[float]:
    """Get embedding for text using OpenAI."""
    if not client:
        return []
    try:
        response = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return []

async def search_similar_documents(query: str, limit: int = 5, channel_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search for similar documents in Milvus."""
    global milvus_client
    if not milvus_client:
        return []
        
    try:
        # Get query embedding
        query_embedding = await get_embedding(query)
        if not query_embedding:
            return []

        # Build filter if channel_name is provided
        filter_expr = None
        if channel_filter:
            filter_expr = f'channel_name == "{channel_filter}"'
        
        # Search in Milvus using MilvusClient
        results = milvus_client.search(
            collection_name=COLLECTION_NAME,
            data=[query_embedding],
            limit=limit,
            filter=filter_expr,
            output_fields=["text", "metadata", "channel_name"]
        )
        
        sources = []
        for hits in results:
            for hit in hits:
                try:
                    # Parse metadata if it's a JSON string
                    metadata = hit['entity'].get('metadata', '{}')
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except:
                            metadata = {"raw": metadata}
                    
                    sources.append({
                        "text": hit['entity'].get('text', ''),
                        "metadata": metadata,
                        "channel_name": hit['entity'].get('channel_name', '')
                    })
                except Exception as e:
                    print(f"Error parsing search result: {e}")
                    # Fallback if metadata parsing fails
                    sources.append({
                        "text": hit['entity'].get('text', ''),
                        "metadata": hit['entity'].get('metadata', {}),
                        "channel_name": hit['entity'].get('channel_name', '')
                    })
        
        return sources
        
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []

async def chat_with_gpt(message: str, conversation_history: List[ChatMessage], sources: Optional[List[Dict[str, Any]]] = None) -> str:
    """Chat with GPT using conversation history and optional RAG sources."""
    if not client:
        return "OpenAI API key not configured."
    try:
        # Prepare system message
        system_message = "You are a helpful AI assistant. Provide accurate and helpful responses. If a youtube link in the source, provide that as well with the proper timestamped youtube url with this format: https://youtube.com/watch?v=<id>&t=<time>s"
        if sources:
            # Enhanced context with channel information
            context_parts = []
            for source in sources:
                try:
                    metadata = source.get('metadata', {})
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                    
                    channel_name = metadata.get('channel_name', source.get('channel_name', 'Unknown Channel'))
                    video_title = metadata.get('video_title', 'Unknown Video')
                    youtube_id = metadata.get('youtube_id', '')
                    timestamp = metadata.get('start_time', '')
                    
                    print(channel_name, metadata)
                    # Format timestamp if available
                    if timestamp:
                        minutes = int(timestamp // 60)
                        seconds = int(timestamp % 60)
                        time_str = f"[{minutes:02d}:{seconds:02d}]"
                    else:
                        time_str = ""
                    
                    # Include YouTube ID if available
                    youtube_info = f" (ID: https://www.youtube.com/watch?v={youtube_id}&t={str(round(timestamp))}s)" if youtube_id else ""
                    context_parts.append(f"Source ({channel_name} - {video_title}{youtube_info}): {source['text']}")
                except Exception as e:
                    print(f"Error formatting source: {e}")
                    # Fallback if metadata parsing fails
                    context_parts.append(f"Source: {source['text']}")
            
            context = "\n\n".join(context_parts)
            system_message += f"\n\n<Sources>:\n{context}"
        
        print(system_message)
        # Prepare messages
        messages = [{"role": "system", "content": system_message}]
        
        # Add conversation history
        for msg in conversation_history[-10:]:  # Limit to last 10 messages
            messages.append({"role": msg.role, "content": msg.content})
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Call OpenAI
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error chatting with GPT: {e}")
        return "I apologize, but I'm having trouble processing your request right now."

# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main chat interface."""
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint with RAG integration."""
    try:
        # Search for relevant documents
        sources = await search_similar_documents(request.message)
        
        print(sources)

        # Convert conversation history to ChatMessage objects
        history = [ChatMessage(role=msg.role, content=msg.content) 
                  for msg in request.conversation_history]
        
        # Get AI response
        response = await chat_with_gpt(request.message, history, sources)
        
        return ChatResponse(
            response=response,
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/add-document")
async def add_document(request: AddDocumentRequest):
    """Add a document to the RAG system."""
    global milvus_client
    if not milvus_client:
        raise HTTPException(status_code=500, detail="Milvus not connected")
        
    try:
        # Generate Murmur3 hash of the text as primary key
        # mmh3.hash can return negative numbers, so we use abs() and convert to int64
        text_hash = abs(mmh3.hash(request.text, signed=False))
        
        print(f"Document hash: {text_hash}")
        
        # Check if document already exists
        try:
            existing_docs = milvus_client.query(
                collection_name=COLLECTION_NAME,
                filter=f"id == {text_hash}",
                output_fields=["id"]
            )
            
            if existing_docs and len(existing_docs) > 0:
                return {"message": "Document already exists", "id": text_hash}
        except Exception as query_error:
            # Collection might not exist yet, or query failed - continue with insert
            print(f"Query check failed (might be first document): {query_error}")
        
        # Get embedding
        print("Generating embedding...")
        embedding = await get_embedding(request.text)
        if not embedding:
            raise HTTPException(status_code=500, detail="Failed to generate embedding")
        
        print(f"Embedding generated: {len(embedding)} dimensions")
        
        # Parse metadata
        try:
            json_metadata = json.loads(request.metadata)
        except:
            json_metadata = {}
        
        # Extract channel_name from metadata or use default
        channel_name = json_metadata.get('channel_name', 'default')
        
        print(f"Metadata: {json_metadata}")
        print(f"Channel name: {channel_name}")

        # Insert into Milvus using MilvusClient
        # MilvusClient.insert expects data as a list of dictionaries
        insert_data = [{
            "id": text_hash,
            "channel_name": channel_name,
            "text": request.text,
            "embedding": embedding,
            "metadata": request.metadata
        }]
        
        print(f"Inserting document with ID: {text_hash}")
        milvus_client.insert(
            collection_name=COLLECTION_NAME,
            data=insert_data
        )
        
        print("Document inserted successfully")
        
        return {"message": "Document added successfully", "id": text_hash}
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log the full error for debugging
        import traceback
        error_details = traceback.format_exc()
        print(f"Error adding document: {error_details}")
        raise HTTPException(status_code=500, detail=f"Failed to add document: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    global milvus_client
    return {"status": "healthy", "milvus_connected": milvus_client is not None}

@app.get("/rag-status")
async def rag_status():
    """Check if RAG/Milvus is available"""
    global milvus_client
    
    # Try to reconnect if not connected
    if not milvus_client:
        connect_to_milvus()
    
    # Test connection by trying to list collections
    is_available = False
    if milvus_client:
        try:
            milvus_client.list_collections()
            is_available = True
        except Exception as e:
            print(f"RAG status check failed: {e}")
            is_available = False
    
    return {
        "available": is_available,
        "status": "connected" if is_available else "disconnected",
        "milvus_uri": MILVUS_URI if MILVUS_URI else "not set",
        "has_token": bool(MILVUS_TOKEN)
    }

# Legacy endpoints for backward compatibility
@app.post("/ask")
async def ask_legacy(request: ChatRequest):
    """Legacy /ask endpoint for backward compatibility."""
    chat_response = await chat(request)
    return {"response": chat_response.response}

@app.post("/add-document")
async def add_document_legacy(request: AddDocumentRequest):
    """Legacy /add-document endpoint for backward compatibility."""
    return await add_document(request)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
