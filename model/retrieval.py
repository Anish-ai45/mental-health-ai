from datasets import load_dataset
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# Load your Hugging Face dataset
dataset = load_dataset("marmikpandya/mental-health", split="train")

# Initialize SentenceTransformer model for generating embeddings
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Extract the 'input' field to create embeddings and 'output' for responses
inputs = dataset['input']
outputs = dataset['output']

# Generate embeddings for the 'input' documents
input_embeddings = embedding_model.encode(inputs)

# Set up FAISS index
dimension = input_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(np.array(input_embeddings))

def retrieve_relevant_response(query, top_k=3):
    # Generate embedding for the query
    query_embedding = embedding_model.encode([query])
    
    # Retrieve top_k similar inputs
    _, indices = faiss_index.search(query_embedding, top_k)
    
    # Get corresponding outputs as responses
    relevant_responses = [outputs[idx] for idx in indices[0]]
    
    return relevant_responses

# # Example usage
# query = "I feel really overwhelmed with my relationship and don't know what to do."
# relevant_responses = retrieve_relevant_response(query)
# print(relevant_responses)
