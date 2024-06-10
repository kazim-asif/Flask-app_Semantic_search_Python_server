# -*- coding: utf-8 -*-

from flask_cors import CORS
from flask import Flask, request, jsonify
import requests
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from googletrans import Translator

chroma_client = chromadb.PersistentClient('chromadb/')
collection = chroma_client.get_or_create_collection(
        name="products",
        metadata={"hnsw:space": "cosine"} # l2 is the default
    )

def readCSVAndGetDocs():
    
    # Read the CSV file into a DataFrame
    selected_data = pd.read_csv('products_data/AllProductswithuid.csv')
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
        
def addDocsFromCsv():
    
    # Read the CSV file into a DataFrame
    selected_data = pd.read_csv('products_data/AllProductswithuid.csv')
    #subset_data = selected_data.head(50)
    
    # Set the batch size (number of rows to add in each iteration)
    batch_size = 500
    
    # Iterate over every alternate batch of 500 rows and add documents
    for i in range(0, len(selected_data), batch_size * 2):
        batch_data = selected_data.iloc[i:i + batch_size]
        
        documents = batch_data['name'].tolist() #getting name column from csv
        ids = batch_data['uid'].tolist() #getting uid column from csv
        
        #adding to db collecction
        collection.add(documents=documents, ids=ids)
    
    """# Iterate over rows and add documents one by one
    for index, row in selected_data.iterrows():
        document = row['name']
        uid = row['uid']
        collection.add(documents=[document], ids=[uid])"""


app = Flask(__name__)

@app.route('/add', methods=['POST'])
def receive_data():
    product = request.json
    product_id = product.get('uid')
    product_name = product.get('name')
    
    # add product to vector db
    collection.add(
        documents = [product_name],
        ids = [product_id]
    )
    return jsonify({"message": "Data added successfully"})

@app.route('/update', methods=['PUT'])
def update_data():
    product = request.json
    product_id = product.get('uid')
    product_name = product.get('name')
    # add product to vector db
    collection.update(
        documents = [product_name],
        ids = [product_id]
    )
    return jsonify({"message": "Data updated successfully"})

# search in database
@app.route('/search', methods=['GET'])
def get_data():
    # Access query parameters using request.args
    search_query = request.args.get('prod', '')
    if search_query:
        results = collection.query(
            query_texts = [search_query],
            n_results=80
        )
        results = results['ids'][0]
    else:
        results = collection.get()
        results = results['ids']
        
    sample_data = {"products": results}
    return jsonify(sample_data)


@app.route("/", methods=["GET"])
def base():
    return jsonify({"message":"i am base route"})

#delete record from database
@app.route('/delete/<int:item_id>', methods=['DELETE'])
def delete_item(item_id):
    item_id = str(item_id) # Convert item_id to a string
    if collection.get(ids=[item_id]):
        collection.delete(ids=[item_id])
    return jsonify({"message": "Data deleted successfully"})

def translate_to_english(text):
    if text:
        translator = Translator()
        translation = translator.translate(text, src='ur', dest='en')
        return translation.text
    else:
        return "No text to translate"

def process_audio(data):
    
    API_URL = "https://api-inference.huggingface.co/models/kazimAsif/whisper-STT-small-ur"
    headers = {"Authorization": "Bearer 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"}
    
    response = requests.post(API_URL, headers=headers, data=data)
    transcribed_data = response.json()

    # Extract the 'text' key from the transcribed_data dictionary
    transcribed_text = transcribed_data.get('text', '')

    # Translate the transcribed text to English
    translated_text = translate_to_english(transcribed_text)
    return translated_text


@app.route('/audio/', methods=['POST'])
def getAudio():
    
    try:
        # Get the audio file
        audio_file = request.files['audio']
        
        # Read the file content into memory
        data = audio_file.read()

        # Process the audio file
        result = process_audio(data)

        return jsonify({"text":result})
    except Exception as e:
        return jsonify({"error": str(e)})
    

def createDescription(prompt):
    API_URL = "https://api-inference.huggingface.co/models/mindthebridge/short-description-generator-6k"
    headers = {"Authorization": "Bearer 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"}
    
    payload = {
        "inputs": prompt,
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    # Check if the request was successful
    if response.status_code == 200:
        return response.json()  # Return JSON data from the response
    else:
        return None  # Return None if request was not successful

    
@app.route('/generateDesc/', methods=['POST'])
def getDescription():
    try:
        prompt = request.form.get('data')
        result = createDescription(prompt)
        
        # Check if the result is not None
        if result is not None:
            return jsonify({"description": result[0]['summary_text']})
        else:
            return jsonify({"error": "Failed to generate description"})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    
    CORS(app) # enable Cross-Origin Resource Sharing
    
    if collection.count()<=0:
        addDocsFromCsv()
    print(collection.count())
    
    app.run(port=5000)  # Choose any available port

