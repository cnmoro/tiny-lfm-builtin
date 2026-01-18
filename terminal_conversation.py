import tiny_lfm_builtin
import os

# 1. Initialize
base_path = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(base_path, "model-q4.gguf")
model = tiny_lfm_builtin.LiquidLFM(weights_path)

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
    streamer = model.generate(history)

    for token_text in streamer:
        print(token_text, end="", flush=True)
        full_response += token_text

    # Append Assistant Message to History
    history.append({"role": "assistant", "content": full_response})
    print("") # Newline