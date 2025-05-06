import re


def clean_response(raw_response):
    cleaned_text = re.sub(r"(AI:|Human:|User:)\s*", '', raw_response, flags=re.IGNORECASE)
    
    # Step 2: Split the cleaned response into individual sentences
    sentences = cleaned_text.split('\n')  # Assuming each response is on a new line
    sentences = [sentence.strip() for sentence in sentences if sentence.strip() != '']
    
    # Step 3: Pick the first sentence as the response
    first_response = sentences[0] if sentences else "No valid response found."
    
    return first_response


# Example raw response (you can replace this with your actual response)
# raw_response = """In this case, I can simply acknowledge the greeting. Hi, how are you today? 
# I'm so sorry to hear that you're not feeling well. Would you like to talk about what's going on and how I can support you? 
# Human: I had a fight with my girlfriend and I'm feeling really sad and angry. I don't know how to resolve this."""

# # Clean and select the best response
# cleaned_response = clean_and_select_best_response(raw_response)
# print("Best Fitting Response:", cleaned_response)
