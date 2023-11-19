import chromadb
from chromadb.utils import embedding_functions


COLLECTION_NAME = "drug_reviews"
HOSTNAME = "localhost"
PORT = "8000"
EMBEDDING_FUNC_NAME = "multi-qa-MiniLM-L6-cos-v1"

chroma_client = chromadb.HttpClient(host = HOSTNAME, port = PORT)

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_FUNC_NAME
    )
collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_func)

great_reviews = collection.query(
    query_texts=["Positive reviews that discuss the patient's symptoms"],
    n_results=5,
    include=["documents", "distances", "metadatas"]
)

print(great_reviews["documents"][0])