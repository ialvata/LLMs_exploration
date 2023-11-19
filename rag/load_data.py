import chromadb
from chromadb.utils import embedding_functions
from chroma_util import create_chroma_collection
from etl import load_car_reviews, load_drug_reviews

COLLECTION_NAME = "edmund_car_reviews"
# COLLECTION_NAME = "drug_reviews"
HOSTNAME = "localhost"
PORT = "8000"
EMBEDDING_FUNC_NAME = "multi-qa-MiniLM-L6-cos-v1"
chroma_client = chromadb.HttpClient(host = HOSTNAME, port = PORT)
collection = create_chroma_collection(COLLECTION_NAME,EMBEDDING_FUNC_NAME,chroma_client = chroma_client)
load_car_reviews(collection)
# load_drug_reviews(collection)