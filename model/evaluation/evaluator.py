import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import matplotlib.pyplot as plt
from datasets import load_dataset
from model.rag_handler import OllamaEmbeddingsWithMemory
from sentence_transformers import SentenceTransformer

class ChatbotEvaluator:
    def __init__(self, model_handler, test_dataset):
        self.model_handler = model_handler
        self.contexts = test_dataset["Context"]
        self.responses = test_dataset["Response"]
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  
    
    def evaluate_relevance(self, predicted_response, ground_truth_response):
        """Evaluate relevance using cosine similarity of embeddings."""
        predicted_embedding = self.model.encode(predicted_response)
        ground_truth_embedding = self.model.encode(ground_truth_response)
        
        # Calculate cosine similarity between the embeddings
        similarity_score = cosine_similarity([predicted_embedding], [ground_truth_embedding])[0][0]
        return similarity_score

    def evaluate_empathy(self, predicted_response):
        """Evaluate empathy using sentiment analysis with adjusted scale."""
        sentiment_score = self.sentiment_analysis(predicted_response)
        if sentiment_score < 0:
            return 0  # Treat negative sentiment as 0 empathy for evaluation
        else:
            return sentiment_score


    def sentiment_analysis(self, text):
        """Perform sentiment analysis using TextBlob."""
        blob = TextBlob(text)
        return blob.sentiment.polarity  # Polarity between -1 (negative) and 1 (positive)

    def evaluate_conciseness(self, predicted_response):
        """Evaluate conciseness by checking text length (word count)."""
        word_count = len(predicted_response.split())
        conciseness_score = 100 - word_count  # Lower word count = higher conciseness score (out of 100)
        return max(0, conciseness_score)  # Ensure score does not go negative

    def evaluate(self, num_samples=10):
        """Evaluate the chatbot for relevance, empathy, and conciseness."""
        relevance_scores = []
        empathy_scores = []
        conciseness_scores = []

        for i in range(num_samples):
            user_query = self.contexts[i]
            ground_truth_response = self.responses[i]
            predicted_response = self.model_handler.get_response(user_query)
            
            relevance_score = self.evaluate_relevance(predicted_response, ground_truth_response)
            empathy_score = self.evaluate_empathy(predicted_response)
            conciseness_score = self.evaluate_conciseness(predicted_response)

            relevance_scores.append(relevance_score)
            empathy_scores.append(empathy_score)
            conciseness_scores.append(conciseness_score)

            print(f"Sample {i + 1}:")
            print(f"Predicted Response: {predicted_response}")
            print(f"Relevance Score: {relevance_score:.2f}, Empathy Score: {empathy_score:.2f}, Conciseness Score: {conciseness_score}")
            print()
        
        # Calculate combined average scores
        avg_relevance = np.mean(relevance_scores)
        avg_empathy = np.mean(empathy_scores)
        avg_conciseness = np.mean(conciseness_scores)

        return avg_relevance, avg_empathy, avg_conciseness, relevance_scores, empathy_scores, conciseness_scores


# Example Usage
test_dataset = load_dataset("Amod/mental_health_counseling_conversations", split="train").select(range(20))

# Assuming `model_handler` is your chatbot handler with a `get_response` method
ollama_handler = OllamaEmbeddingsWithMemory(model_name="mental_health_ai")
chatbot_evaluator = ChatbotEvaluator(model_handler=ollama_handler, test_dataset=test_dataset)
avg_relevance, avg_empathy, avg_conciseness, relevance_scores, empathy_scores, conciseness_scores = chatbot_evaluator.evaluate(num_samples=11)

print(f"Average Relevance: {avg_relevance:.2f}")
print(f"Average Empathy: {avg_empathy:.2f}")
print(f"Average Conciseness: {avg_conciseness:.2f}")

# Plotting the results
def plot_scores(relevance_scores, empathy_scores, conciseness_scores):
    samples = range(1, len(relevance_scores) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(samples, relevance_scores, label='Relevance', marker='o')
    plt.plot(samples, empathy_scores, label='Empathy', marker='o')
    plt.plot(samples, conciseness_scores, label='Conciseness', marker='o')
    
    plt.xlabel('Sample Number')
    plt.ylabel('Score')
    plt.title('Chatbot Evaluation: Relevance, Empathy, Conciseness')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_scores(relevance_scores, empathy_scores, conciseness_scores)
