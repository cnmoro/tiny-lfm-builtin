from tiny_lfm import TinyLFM
import os

# 1. Initialize
model = TinyLFM(model_size="350M")

# 2. State Management
history = []

print("--- Liquid LFM2 Chat (Type 'quit' to exit) ---")

while True:
    user_input = input("\nUser: ")
    if user_input.lower() in ["quit", "exit"]:
        break

    # Append User Message
    history.append({"role": "user", "content": user_input})

    # Prepare for streaming response
    print("Assistant: ", end="", flush=True)
    full_response = ""
    
    # Pass the WHOLE history to Rust
    # Rust will clear cache -> prefill history -> generate new tokens
    streamer = model.chat(history)

    for token_text in streamer:
        print(token_text, end="", flush=True)
        full_response += token_text

    # Append Assistant Message to History
    history.append({"role": "assistant", "content": full_response})
    print("") # Newline