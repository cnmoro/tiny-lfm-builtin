from tiny_lfm import TinyLFM
import time

# Initialize model via Python wrapper
model = TinyLFM()

# Define multiple chat sessions
chats = [
    [
        {"role": "user", "content": "Hello, what is the capital of France?"}
    ],
    [
        {"role": "system", "content": "You are a pirate."},
        {"role": "user", "content": "Tell me a joke."}
    ],
    [
        {"role": "user", "content": "Write a python function to add two numbers."}
    ]
]

print(f"--- Processing {len(chats)} chats in parallel ---")
start_time = time.time()

# This uses the stateless parallel implementation through the wrapper
results = model.batch_chat(chats, max_tokens=50)

end_time = time.time()
print(f"Total time: {end_time - start_time:.2f}s")
print(f"Time per chat: {(end_time - start_time) / len(chats):.2f}s")

for i, r in enumerate(results):
    print(f"\n--- Chat {i+1} Output ---")
    # Result contains the full text including prompt. 
    # Since we don't return the raw prompt length from Rust, 
    # we just print the tail or the whole thing.
    print(r[-200:]) # Printing last 200 chars for brevity