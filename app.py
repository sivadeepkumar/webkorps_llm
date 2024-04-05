from flask import Flask, request, jsonify, g
from model_service import *
from dotenv import load_dotenv
from flask_cors import CORS 
import os
import logging
load_dotenv()

# Configure logging
logging.basicConfig(filename='./logs/ai_model.log', level=logging.INFO)
logger = logging.getLogger(__name__)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health_check():
    return 'OK', 200


@app.route('/cryoport', methods=['POST'])
def cryoport():
    try:
        data = request.get_json()
        query = data['query']
        processed_data = process_query('cryoport_text.txt', query)

        return jsonify(processed_data)
    except Exception as e:
        logger.exception("An error occurred in /cryoport endpoint."+str(e))
        return jsonify({'Error': 'Internal Server Error'}), 500

@app.route('/realEstateQuery', methods=['POST'])
def realEstateQuery():
    try:
        data = request.get_json()
        query = data['query']
        processed_data = process_query('estate.txt', query)

        return jsonify(processed_data)
    except Exception as e:
        logger.exception("An error occurred in /realEstateQuery endpoint.")
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/query', methods=['POST'])
def assetpanda():
    try:
        data = request.get_json()
        query = data['query']
        processed_data = process_query('assetpanda.txt', query)

        return jsonify(processed_data)
    except Exception as e:
        logger.exception("An error occurred in /query (ASSETPANDA) endpoint.")
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/webkorps_query', methods=['POST'])
def webkorps_query():
    try:
        data = request.get_json()
        query = data['query']
        processed_data = process_query('webkorps_data.txt', query)

        return jsonify(processed_data)
    except Exception as e:
        logger.exception("An error occurred in /webkorps_query endpoint.")
        return jsonify({'error': 'Internal Server Error'}), 500


@app.route('/summary', methods=['POST'])
def summary():
    data = request.get_json()
    query = data['query']
    source = data['source']


    embeddings = OpenAIEmbeddings()  
    document_search = FAISS.from_texts([source], embeddings)
    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    docs = document_search.similarity_search(query)
    result = chain.run(input_documents=docs, question=query)

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)