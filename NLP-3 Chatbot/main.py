# Install (run once)
!pip install transformers torch

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model
model_name = "microsoft/DialoGPT-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Start chatbot (same line format)
print("Chatbot: Hello! I am your AI assistant. How can I help you today?", end=" ")

chat_history_ids = None

while True:
    user_input = input("User: ")

    # Exit condition (NO message after exit)
    if user_input.lower() in ["exit", "quit"]:
        break

    # -------------------------------
    # RULE-BASED IMPROVEMENTS (for accuracy)
    # -------------------------------

    if user_input.lower() in ["hello", "hi", "hey"]:
        print("Chatbot: Hello! Nice to meet you. How can I assist you today?", end=" ")
        continue

    if "thank" in user_input.lower():
        print("Chatbot: You're welcome! Feel free to ask more questions.", end=" ")
        continue

    if "who created python" in user_input.lower():
        print("Chatbot: Python was created by Guido van Rossum and released in 1991.", end=" ")
        continue

    if "artificial intelligence" in user_input.lower() or "what is ai" in user_input.lower():
        print("Chatbot: Artificial Intelligence refers to the simulation of human intelligence by machines that can perform tasks such as learning, reasoning, and problem solving.", end=" ")
        continue

    # -------------------------------
    # TRANSFORMER MODEL RESPONSE
    # -------------------------------

    # Improve response quality
    user_input = "Respond clearly and correctly: " + user_input

    # Tokenize input
    new_input = tokenizer(user_input + tokenizer.eos_token, return_tensors='pt')
    input_ids = new_input["input_ids"]

    # Maintain conversation history
    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1)
    else:
        bot_input_ids = input_ids

    # Generate response
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=500,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=40,
        top_p=0.9,
        temperature=0.5
    )

    # Decode response
    response = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    # Print in required format
    print("Chatbot:", response, end=" ")
