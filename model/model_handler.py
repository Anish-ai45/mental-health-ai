from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory

class ModelHandler:
    def __init__(self, model_name):
        # Initialize the language model and memory
        self.llm = OllamaLLM(model=model_name)
        self.memory = ConversationBufferMemory()

    def get_response(self, user_input, context):
        # Update memory with the latest user input
        self.memory.save_context({"input": user_input}, {"output": ""})

        # Create the post-prompt including context
        post_prompt =  (
            f"Context: {context}\n"
            "Respond in a concise, empathetic, and compassionate manner. "
            "Avoid repeating or rephrasing the user's input. "
            "Provide helpful support and suggestions without giving long or unnecessary explanations. "
            "Be mindful of the user's feelings and emotional state, keeping the tone supportive and non-judgmental."
        )

        # Build messages for the model
        messages = [
            {"role": "user", "content": user_input},
            {"role": "system", "content": post_prompt},
        ]

        try:
            # Get response from Ollama using invoke
            response = self.llm.invoke(messages)

            # Extract and format the model response
            model_response = response.get('message', {}).get('content', '').strip() if not isinstance(response, str) else response.strip()

            # Clean up the response to remove any unwanted text like "AI:", "Human:", "System:", or "User:"
            model_response = model_response.replace("AI:", "").replace("Human:", "").replace("System:", "").replace("User:", "").strip()

            # Remove the system message part if it still exists
            if "System:" in model_response:
                model_response = model_response.split("System:")[-1].strip()

            # Return the cleaned up response
            return model_response
        
        except Exception as e:
            raise Exception(f'Error contacting the model: {str(e)}')
