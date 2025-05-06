import sys
import os
print(sys.path)
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))


from langchain.memory import ConversationBufferMemory
from llama_cpp import Llama
from model.clean_ai_response import clean_response  
from model.rag_handler import retrieve_context

# Global variables to store the model and memory
llm = None
memory = None

def initialize_model():
    global llm, memory
    
    llm = load_model()
    memory = ConversationBufferMemory()
    
    if not llm:
        print("Error loading the model. Exiting.")
        return False
    
    print("Model initialized successfully.")
    return True


# Load the model
def load_model(repo_id="Anish45/mental-health-model-GGUF", filename="unsloth.Q5_K_M.gguf"):
    try:
        llm_instance = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_ctx=512,        # Reduced context window
            n_threads=4,      # Controlled thread count
            n_gpu_layers=10,  # Full GPU acceleration
            low_vram=True,    # Memory optimization
            verbose=False
        )
        return llm_instance
    except Exception as e:
        print(f"Model loading error: {e}")
        return None



# Generate prompt with retrieved context
def generate_prompt(user_query, threshold=0.5):
    global memory
    
    # Retrieve past conversation history
    conversation_history = memory.load_memory_variables({})["history"] if memory else ""

    # Retrieve context with similarity score
    retrieved_context = retrieve_context(user_query)
    
    valid_context = ""
    
    # Filter the retrieved context based on similarity score threshold
    for item in retrieved_context:
        if item["similarity_score"] >= threshold:
            valid_context += f"{item['context']}\n{item['response']}\n"

    # Mental health assistant prompt
    mental_health_prompt = (
        "You are a compassionate mental health assistant. Always respond empathetically and provide a supportive, brief reply. "
        "If the user shares distressing or sensitive information, acknowledge their feelings, provide support, and gently ask follow-up questions. "
        "Use the conversation history to ensure continuity and context awareness. "
        "If the user is in crisis, suggest they seek immediate help from a professional. "
        "For casual greetings, respond warmly but do not repeat affirmations unnecessarily. "
        "Remember to provide support and suggest resources when appropriate. "
        "Never include instructions, examples, or meta-text in your response. Your replies must remain focused on mental health and be concise and empathetic."
        "Under no circumstances should you generate code, technical explanations, or non-mental-health advice. "
        "Your response should be concise and supportive."
    )

    
    # Construct the final prompt with conversation history, valid context, and user query
    prompt = f"{mental_health_prompt}\n\nConversation History:\n{conversation_history}\n\n"
    if valid_context:
        prompt += f"Context: {valid_context}\n"

    prompt += f"User Query: {user_query}\nAssistant:"
    
    return prompt




def generate_response(user_query, max_tokens=50):
    global llm, memory

    try:
        if llm is None:
            print("Model not loaded properly. llm is None.")
            return "Model is not initialized correctly."
        
        # Generate the prompt with the relevant context
        prompt = generate_prompt(user_query)
        
        # Generate the response from the model
        response = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.6,
            top_p=0.9,
            stop=["User:", "AI:"]
        )
        
        response_text = response['choices'][0]['text'].strip()

        cleaned_response = clean_response(response_text)
        # Save the response to memory
        memory.save_context({"input": user_query}, {"output": cleaned_response})
        print("Final Cleaned Response", cleaned_response)
        return cleaned_response
        
      
            
               
    except Exception as e:
        print(f"Response generation error: {e}")
        return "Unable to generate response"




# Main usage example
# if __name__ == "__main__":
#     if initialize_model():
#         while True:
#             user_input = input("User: ")
#             if user_input.lower() in ["exit", "quit"]:
#                 print("Exiting the chat. Take care!")
#                 break

#             # Call generate_response with only the user input
#             response = generate_response(user_input)

#             # Display the response
#             print(f"AI: {response}")
