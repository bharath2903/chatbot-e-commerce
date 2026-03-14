import os

import pandas as pd
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
groq_client = Groq()
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

def faq_chain(result):
    context = " ".join([r.get('answer') for r in result['metadatas'][0]])
    return context
def generate_answer(query, context):
    prompt = f''' 
    Given the question and context below, come up with a 
    natural easy to understand response for a customer who has an issue using only the context mentioned. If you dont find the 
    answer based on the below context , tell" I am not sure ,our team will contact you soon for assistance". Please do not 
    make things up
    Question: {query}
    
    Context: {context}
    '''
    chat_completion = groq_client.chat.completions.create(

        messages=[
            {
                "role": "user",
                "content":prompt,
    }
    ],
    model = os.environ['GROQ_MODEL'],
    )
    return chat_completion.choices[0].message.content






if __name__ =='__main__':
    ingest_faq(faq_path)
    query='Do i always need to do online payments , or can i use cash also??'
    result = get_relavent_qa(query)
    context = faq_chain(result)
    answer = generate_answer(query,context)
    print(answer)