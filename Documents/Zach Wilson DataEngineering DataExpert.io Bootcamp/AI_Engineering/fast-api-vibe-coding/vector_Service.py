import os
import hashlib
from typing import List, Optional, Dict, Any
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
COLLECTION_NAME = "rag_documents"
EMBEDDING_DIM = 1536  # OpenAI text-embedding-ada-002 dimension
EMBEDDING_MODEL = "text-embedding-ada-002"


class VectorService:
    """Service for managing vector database operations with Milvus"""
    
    def __init__(self):
        self.collection = None
        self.openai_client = None
        # Don't initialize OpenAI on startup - do it lazily when needed
    
    def _initialize_openai(self):
        """Initialize OpenAI client"""
        if self.openai_client is not None:
            return
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.")
        
        # Initialize with only api_key - OpenAI 1.3.0 doesn't support proxies parameter
        # If you need proxies, set HTTP_PROXY/HTTPS_PROXY environment variables instead
        self.openai_client = OpenAI(api_key=api_key)
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI"""
        if not self.openai_client:
            self._initialize_openai()
        
        response = self.openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    
    def _init_milvus_collection(self) -> Collection:
        """Initialize Milvus connection and collection (supports both local and managed Milvus/Zilliz)"""
        # Check for managed Milvus (Zilliz Cloud) URI first
        milvus_uri = os.getenv("MILVUS_URI")
        milvus_token = os.getenv("MILVUS_TOKEN")  # For Zilliz Cloud authentication
        
        if milvus_uri:
            # Managed Milvus (Zilliz Cloud) connection
            try:
                if milvus_token:
                    connections.connect(
                        alias="default",
                        uri=milvus_uri,
                        token=milvus_token,
                        timeout=2
                    )
                else:
                    connections.connect(
                        alias="default",
                        uri=milvus_uri,
                        timeout=2
                    )
            except Exception as e:
                raise ConnectionError(f"Failed to connect to managed Milvus. Error: {str(e)}")
        else:
            # Local Milvus connection
            milvus_host = os.getenv("MILVUS_HOST", "localhost")
            milvus_port = os.getenv("MILVUS_PORT", "19530")
            
            # Quick check if port is open before attempting connection (very fast)
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.1)  # 100ms timeout - very fast check
            result = sock.connect_ex((milvus_host, int(milvus_port)))
            sock.close()
            
            if result != 0:
                raise ConnectionError(f"Milvus is not running at {milvus_host}:{milvus_port}")
            
            # Connect to local Milvus
            try:
                connections.connect(
                    alias="default",
                    host=milvus_host,
                    port=milvus_port,
                    timeout=2  # 2 second connection timeout
                )
            except Exception as e:
                raise ConnectionError(f"Failed to connect to Milvus at {milvus_host}:{milvus_port}. Error: {str(e)}")
        
        # Check if collection exists, if not create it
        if not utility.has_collection(COLLECTION_NAME):
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535)
            ]
            schema = CollectionSchema(fields, "RAG documents collection")
            
            # Create collection
            collection = Collection(COLLECTION_NAME, schema)
            
            # Create index on embedding field
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            collection.create_index("embedding", index_params)
        else:
            collection = Collection(COLLECTION_NAME)
        
        # Load collection to memory
        collection.load()
        return collection
    
    def get_collection(self) -> Collection:
        """Get or initialize Milvus collection"""
        if self.collection is None:
            self.collection = self._init_milvus_collection()
        return self.collection
    
    def add_document(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a document to the vector database
        
        Args:
            text: The text content of the document
            metadata: Optional metadata dictionary
            
        Returns:
            The document ID
        """
        try:
            # Generate embedding
            embedding = self.get_embedding(text)
            
            # Generate unique ID for the document
            doc_id = hashlib.md5(text.encode()).hexdigest()
            
            # Prepare metadata
            metadata_str = str(metadata) if metadata else "{}"
            
            # Get collection
            collection = self.get_collection()
            
            # Insert document
            collection.insert([
                [doc_id],
                [text],
                [embedding],
                [metadata_str]
            ])
            
            # Flush to ensure data is written
            collection.flush()
            
            return doc_id
        except Exception as e:
            raise Exception(f"Failed to add document: {str(e)}")
    
    def search(self, query: str, limit: int = 3) -> List[str]:
        """
        Search for similar documents in the vector database
        
        Args:
            query: The search query text
            limit: Number of results to return (default: 3)
            
        Returns:
            List of text content from similar documents
        """
        try:
            # Get query embedding
            query_embedding = self.get_embedding(query)
            
            # Get collection
            collection = self.get_collection()
            
            # Search parameters
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            
            # Perform search
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                output_fields=["text", "metadata"]
            )
            
            # Extract relevant context from search results
            context_parts = []
            if results and len(results[0]) > 0:
                for hit in results[0]:
                    context_parts.append(hit.entity.get("text"))
            
            return context_parts
        except Exception as e:
            raise Exception(f"Vector search failed: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if Milvus is available and connected"""
        try:
            self.get_collection()
            return True
        except Exception:
            return False


# Global instance
_vector_service = None


def get_vector_service() -> VectorService:
    """Get or create the global VectorService instance"""
    global _vector_service
    if _vector_service is None:
        _vector_service = VectorService()
    return _vector_service

