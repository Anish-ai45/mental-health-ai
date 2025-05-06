from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu
import spacy
from model.rag_handler import OllamaEmbeddingsWithMemory

# Load the evaluation dataset
eval_dataset = load_dataset("Amod/mental_health_counseling_conversations", split="train").select(range(50))

# Initialize the chatbot
chatbot = OllamaEmbeddingsWithMemory(model_name="mental_health_ai")

# Evaluate the chatbot's responses
for user_input, expected_response in eval_dataset:
    response = chatbot.get_response(user_input)
    
    # Calculate BLEU score
    bleu_score = sentence_bleu([expected_response], response)
    print(f"BLEU Score: {bleu_score:.4f}")
    
    # Evaluate empathy using spaCy
    nlp = spacy.load("en_core_web_sm")
    expected_response_doc = nlp(expected_response)
    response_doc = nlp(response)
    empathy_score = ...
    print(f"Empathy Score: {empathy_score:.4f}")