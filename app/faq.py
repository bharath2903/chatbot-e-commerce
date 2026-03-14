import pandas as pd
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq


chroma_client = chromadb.Client()
collection_name_faq = 'faqs'
faq_path = Path(__file__).parent / 'resources/faq_data.csv'
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='sentence-transformers/all-MiniLM-L6-v2')

def ingest_faq(path):
    if collection_name_faq not in [c.name for c in chroma_client.list_collections()]:
        print("Ingestion of data into chromaDB has began...")
        collection = chroma_client.get_or_create_collection(
            name=collection_name_faq,
            embedding_function =ef
        )



        df = pd.read_csv(path)
        docs = df['question'].tolist()
        metadata = [{'answer':ans }for ans in df['answer'].to_list()]
        ids = [f'id_{ids}'for ids in range(len(docs))]
        collection.add(
            documents=docs,
            metadatas=metadata,
            ids = ids
        )
    else:
        print(f"Collection {collection_name_faq} already exists!")

def get_relavent_qa(query):
    collection = chroma_client.get_collection(name=collection_name_faq)
    result = collection.query(
        query_texts=[query],
        n_results=3
    )
    return result



if __name__ =='__main__':
    ingest_faq(faq_path)
    query='how do i return an item?'
    print(get_relavent_qa(query))