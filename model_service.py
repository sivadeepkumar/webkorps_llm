# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS 
from langchain_community.llms.openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
import logging

logging.basicConfig(filename='./logs/model_service.log', level=logging.INFO)
logger = logging.getLogger(__name__)

def process_query(file_name, query):
    try:
        # Initialize embeddings and load documents
        embeddings = OpenAIEmbeddings()
        root_path = './sample_data/text_samples/'
        full_path = root_path + file_name
        with open(full_path, 'r') as f:
            texts = f.read()
            
        # Perform FAISS search
        searched_documents = FAISS.from_texts([texts], embeddings)
        # Load QA chain model
        model_chain = load_qa_chain(OpenAI(), chain_type="stuff")
        # Perform similarity search
        docs = searched_documents.similarity_search(query)
        # Run the QA chain model
        result = model_chain.run(input_documents=docs, question=query)
        
        return result
    except Exception as e:
        logger.exception("Error in processing query."+str(e)+"file_name: "+file_name+"query: "+query)
