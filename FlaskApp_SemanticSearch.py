# -*- coding: utf-8 -*-


from flask import Flask, request, jsonify
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

chroma_client = chromadb.PersistentClient('FYP/chromadb')
collection = chroma_client.get_or_create_collection(
        name="products",
        metadata={"hnsw:space": "cosine"} # l2 is the default
    )

def readCSVAndGetDocs():
    
    # Read the CSV file into a DataFrame
    selected_data = pd.read_csv('FYP/products_data/cleaned_data.csv')
    subset_data = selected_data.head(50)
    documents = subset_data['name'].tolist() # Extract only the "name" field from the DataFrame
    return documents

def generateDocsIds(documents):
    # Generate unique IDs for each document
    document_ids = [f"id{i}" for i in range(1, len(documents) + 1)]
    return document_ids

def generateEmbeddings(documents):
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    embeddings = sentence_transformer_ef(documents)
    return embeddings

def addDocsToCollection():
    if not collection.count() > 0:
        documents = readCSVAndGetDocs()
        ids = generateDocsIds(documents)
        collection.add(
            documents = documents,
            ids = ids
        )


app = Flask(__name__)

@app.route('/add', methods=['POST'])
def receive_data():
    product = request.json
    product_id = product.get('id')
    product_name = product.get('name')
    
    # add product to vector db
    collection.add(
        documents = [product_name],
        ids = product_id
    )
    
    return jsonify({"message": "Data added successfully"})

@app.route('/search', methods=['GET'])
def get_data():
    # Access query parameters using request.args
    search_query = request.args.get('prod', '')
    
    results = collection.query(
        query_texts = [search_query],
        n_results=2
    )
    
    sample_data = {"products": results['ids'][0]}
    return jsonify(sample_data)

if __name__ == '__main__':
    addDocsToCollection()
    app.run(port=5000)  # Choose any available port

