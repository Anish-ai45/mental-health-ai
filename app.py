from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from ollama import Client
from model.retrieval import retrieve_relevant_response  # Use the updated retrieval function
from model.model_handler import ModelHandler  # Import the ModelHandler class

app = Flask(__name__)
CORS(app)  # Allow all domains to access your API

# Initialize Ollama client
client = Client(host='http://localhost:11434')

# Initialize the ModelHandler
model_handler = ModelHandler(model_name='mental_health_ai')  # Use your model name

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    user_input = request.json.get('input')

    # Retrieve relevant responses from the knowledge base using the updated retrieval function
    relevant_responses = retrieve_relevant_response(user_input)

    # Limit the number of responses to avoid excessive context length
    max_responses = 3  # You can adjust this number based on your preference
    limited_responses = relevant_responses[:max_responses]

    # Combine the relevant responses with the user input for context
    context = " ".join(limited_responses) + "\nUser Input: " + user_input  # Combining context and input
    print("Context with limited relevant responses:", context)

    try:
        # Get response from the ModelHandler using the combined context
        model_response = model_handler.get_response(user_input, context)

        # Return only the assistant's response
        return jsonify({'response': model_response})

    except Exception as e:
        return jsonify({'response': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
