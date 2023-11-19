import pathlib
import chromadb
from chromadb.utils import embedding_functions
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

def get_create_chroma_collection(
    collection_name: str,
    embedding_func_name: str = "all-MiniLM-L6-v2",
    distance_func_name: str = "cosine",
    chroma_client: ClientAPI | None = None,
    hostname: str = "localhost",
    port: str = "8000",
) -> Collection:
    """Create a ChromaDB collection"""

    if chroma_client is None:
        chroma_client = chromadb.HttpClient(host = hostname, port = port)

    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name = embedding_func_name
    )

    return chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_func,
        metadata={"hnsw:space": distance_func_name},
    )