from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from src.config import Config
from dotenv import load_dotenv

load_dotenv()
COLLECTION_NAME = "med-index"

def ingest_data():
    embeddings = MistralAIEmbeddings(model=Config.EMBED_MODEL, api_key = Config.MISTRALAI_API_KEY)

    loader = PyPDFDirectoryLoader(Config.Directory_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
    chunked_docs = splitter.split_documents(docs)
    print('chunks are created')

    qdrant_client = QdrantClient(
        url=Config.QDRANT_API_URL, 
        api_key=Config.QDRANT_API_KEY,
    )

    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )
    print(f"Collection '{COLLECTION_NAME}' created successfully!")

    qdrant = QdrantVectorStore.from_documents(
        chunked_docs,
        embeddings,
        url= Config.QDRANT_API_URL,
        api_key= Config.QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        timeout = 6000
    )
    print("Data successfully ingested into Qdrant")

    return qdrant

#qdrant = ingest_data() only use when who have to ingest the data for the first time

def get_vector_store():
    embeddings = MistralAIEmbeddings(
    model= Config.EMBED_MODEL,
    api_key = Config.MISTRALAI_API_KEY
    )

    qdrant_client = QdrantClient(
        url= Config.QDRANT_API_URL, 
        api_key=Config.QDRANT_API_KEY,
    )

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )

    return vector_store

if __name__ == "__main__":
    get_vector_store()