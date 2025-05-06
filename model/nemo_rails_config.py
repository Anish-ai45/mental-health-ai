from nemoguardrails import RailsConfig, LLMRails
import os

# Use absolute path to guardrails_config

config = RailsConfig.from_path("/Users/anishilapaka/Documents/mental-health-ai/guardrails_config/config.yml")
rails = LLMRails(config)

def validate_with_guardrails(user_input, model_response):
    messages = [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": model_response}
    ]
    result = rails.generate(messages=messages)
    return result["content"]

response = validate_with_guardrails("Hi", "Hello, how can I support you?")
print("Guardrails output:", response)
