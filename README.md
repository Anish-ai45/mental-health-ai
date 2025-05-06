# 🧠 Mental Health AI Chatbot

This is a mental health support chatbot built using a fine-tuned LLaMA model with retrieval-augmented generation (RAG). The system uses Hugging Face for inference and provides contextual mental health responses.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Anish-ai45/mental-health-ai.git
cd mental-health-ai
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt`
```


### 4. Create `config.env`
```bash
HUGGINGFACEHUB_API_TOKEN=your_huggingface_access_token_here`
```

### 5. Run the App
```bash
python app.py
```
The app will start on:
http://localhost:5000


## :file_folder: Project Structure

```bash
mental-health-ai/
│
├── app.py
├── config.env
├── requirements.txt
├── setup.py
├── guardrails_config/
│   ├── config.yml
│   └── rails/
│       └── mental_health_flows.co
├── model/
│   ├── __init__.py
│   ├── model_handler.py
│   ├── rag_handler.py
│   ├── clean_ai_response.py
│   ├── nemo_rails_config.py
│   └── evaluation/
│       ├── __init__.py
│       ├── eval.py
│       └── evaluator.py
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js

```



> **[!NOTE]  Known Issues**
> 
> Guardrails are currently not functional
> Guardrails setup files are included, but integration is still in progress. Future updates will enable semantic validation and safety filters.



### :mailbox_with_mail: Contact
Maintained by @Anish-ai45.
Feel free to raise an issue or contribute!

:email: anishilapaka45@gmail.com














