from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from model.model_handler import generate_response, initialize_model




app = Flask(__name__)
app.secret_key = 'local_development_key'  
CORS(app)  

if not initialize_model():
    print("Failed to initialize the model. Exiting.")
    exit(1)

@app.route('/')
def home(): 
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    user_input = request.json.get('input')

    try:
        model_response = generate_response(user_input)
        return jsonify({'response': model_response})

    except Exception as e:
        return jsonify({'response': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


