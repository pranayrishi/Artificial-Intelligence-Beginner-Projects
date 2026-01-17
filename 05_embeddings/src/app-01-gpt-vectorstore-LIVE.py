from pathlib import Path
from llama_index import (
    GPTVectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from dotenv import load_dotenv, find_dotenv

# Load environment variables from a .env file if available
load_dotenv(find_dotenv())

DATA_DIR = "../data"
STORAGE_DIR = "storage"


def get_index():
    index = None

    if Path(STORAGE_DIR).exists():
        # If the index exists already, just load it
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(storage_context)
    else:
        # Read data files
        docs = SimpleDirectoryReader(DATA_DIR).load_data()

        # Chunk the local files (or from APIs) & index them
        index = GPTVectorStoreIndex.from_documents(docs)

        # Save the index locally so that we don't need to do this again
        index.storage_context.persist(persist_dir=STORAGE_DIR)

    return index


def get_response(query):
    index = get_index()
    query_engine = index.as_query_engine()
    return query_engine.query(query)


if __name__ == '__main__':
    while True:
        query = input("What question do you have? ('quit' to quit) ")
        if "quit" in query:
            break
        response = get_response(query)
        print(response)
