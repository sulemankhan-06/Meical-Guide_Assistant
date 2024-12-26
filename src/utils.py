from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import ServerlessSpec
from pinecone import Pinecone
from langchain_mistralai import MistralAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from config import Config

index_name = "med-index"


def ingest_data():
    embeddings = MistralAIEmbeddings(model=Config.EMBED_MODEL,
                                     api_key=Config.MISTRALAI_API_KEY)

    loader = PyPDFDirectoryLoader(Config.Directory_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
    chunked_docs = splitter.split_documents(docs)
    print('chunks are created')

    pc = Pinecone(api_key=Config.PINECONE_API_KEY)

    pc.create_index(name=index_name,
                    dimension=1024,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"))

    pinecone = PineconeVectorStore.from_documents(chunked_docs,
                                                  embedding=embeddings,
                                                  index_name=index_name)

    return pinecone


#pinecone = ingest_data() only for the first time for ingesting data


def get_vector_store():
    embeddings = MistralAIEmbeddings(model=Config.EMBED_MODEL,
                                     api_key=Config.MISTRALAI_API_KEY)
    pc = Pinecone(api_key=Config.PINECONE_API_KEY)
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    return vector_store


if __name__ == "__main__":
    get_vector_store()
