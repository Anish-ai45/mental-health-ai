from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from rank_bm25 import BM25Okapi
import numpy as np
import faiss
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"



# Load the dataset
dataset = load_dataset("Amod/mental_health_counseling_conversations", split="train").select(range(100))
contexts = dataset["Context"]
responses = dataset["Response"]

# Prepare documents with chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = []
for context, response in zip(contexts, responses):
    if context and response:
        context_chunks = text_splitter.split_text(f"Context: {str(context)}")
        for chunk in context_chunks:
            documents.append(f"{chunk} Response: {str(response)}")

print(f"Number of Chunks Created: {len(documents)}")

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')

# Generate embeddings with normalization
inputs = tokenizer(documents, padding=True, truncation=True, return_tensors="pt", max_length=512)
with torch.no_grad():
    embeddings = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
    embeddings = normalize(embeddings, axis=1, norm='l2')  

# Initialize FAISS with normalized embeddings
d = embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(d)  
faiss_index.add(embeddings.astype(np.float32))

# Pre-process documents for BM25 (only once)
tokenized_docs = [doc.split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# Initialize the BART summarization pipeline
summarizer = pipeline("summarization", model="t5-small",device=-1)

# Function to summarize a given text (context or response)
def summarize_context(text):
    input_length = len(text.split())  # number of words, or use len(tokenizer.encode(text)) for tokens

    # Set max_length to 60% of input length but cap it to avoid extremely short or long summaries
    max_len = max(15, min(int(input_length * 0.6), 100))  # between 15 and 100 tokens
    min_len = max(5, int(max_len * 0.5))  # at least half of max_len

    summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
    return summary[0]['summary_text']


def calculate_similarity(query_embedding, context_embedding):
    """
    Calculate cosine similarity between the query and context embeddings.
    """
    similarity = cosine_similarity([query_embedding], [context_embedding])
    return similarity[0][0]

def retrieve_context(user_query, k=2, alpha=0.7):
    # Dense retrieval using FAISS
    query_input = tokenizer(user_query, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        query_embedding = model(**query_input).last_hidden_state.mean(dim=1).cpu().numpy()
        query_embedding = normalize(query_embedding, axis=1, norm='l2')

    # FAISS search for dense retrieval
    distances, indices = faiss_index.search(query_embedding, k)

    # Sparse retrieval using BM25
    sparse_scores = bm25.get_scores(user_query.split())

    # Normalization of sparse and dense scores for score fusion
    sparse_scores = np.array(sparse_scores)
    if np.max(sparse_scores) > 0:
        sparse_scores = sparse_scores / np.max(sparse_scores)  
    distances = 1 - (distances / np.max(distances))  

    # Combine scores using alpha blending
    combined_scores = []
    dense_docs = [documents[i] for i in indices[0]]
    for i, doc in enumerate(dense_docs):
        combined_score = alpha * distances[0][i] + (1 - alpha) * sparse_scores[i]
        combined_scores.append((combined_score, doc))

    # Sort by combined score
    combined_scores.sort(key=lambda x: x[0], reverse=True)

    # Extract context and response, then summarize both
    result = []
    for _, doc in combined_scores[:k]:
        context_response = doc.split("Response:")
        if len(context_response) == 2:
            context = context_response[0].replace("Context:", "").strip()
            response = context_response[1].strip()

            # Summarize context and response individually
            summarized_context = summarize_context(context)
            summarized_response = summarize_context(response)

            # Calculate similarity between the query and context
            context_input = tokenizer(context, padding=True, truncation=True, return_tensors="pt", max_length=512)
            with torch.no_grad():
                context_embedding = model(**context_input).last_hidden_state.mean(dim=1).cpu().numpy()
                context_embedding = normalize(context_embedding, axis=1, norm='l2')

            # Calculate the similarity score
            similarity_score = calculate_similarity(query_embedding[0], context_embedding[0])  # Access the 1st dimension
            
            result.append({
                "context": summarized_context,
                "response": summarized_response,
                "similarity_score": similarity_score
            })
    return result


# Example usage
# user_query = "Hi"
# retrieved_context = retrieve_context(user_query)

# # Display summarized context and responses
# print("\nRetrieved Context and Responses:")
# for item in retrieved_context:
#     # Access the context, response, and similarity_score from the dictionary
#     context = item["context"]
#     response = item["response"]
#     similarity_score = item["similarity_score"]

#     print(f"Summarized Context: {context}\nSummarized Response: {response}\nsimilarity_score: {similarity_score}\n")
