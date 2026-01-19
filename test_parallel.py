from tiny_lfm import TinyLFM
import time

# Initialize model via Python wrapper
model = TinyLFM()

# Define multiple prompts
prompts = [
    "The capital of France is",
    "To bake a cake, you first need to",
    "Rust is a programming language that",
    "The fastest animal on earth is",
    "Artificial Intelligence will",
    "1 + 1 equals",
    "Liquid LFM is",
    "The ocean is blue because"
]

print(f"--- Processing {len(prompts)} prompts in parallel ---")
start_time = time.time()

# This uses the stateless parallel implementation through the wrapper
results = model.batch_completion(prompts, max_tokens=30)

end_time = time.time()
print(f"Total time: {end_time - start_time:.2f}s")
print(f"Time per prompt: {(end_time - start_time) / len(prompts):.2f}s")

for p, r in zip(prompts, results):
    # 'r' contains the full text (prompt + generation)
    # Let's strip the prompt to see just the completion
    completion = r[len(p):].strip()
    print(f"\n[Prompt]: {p}\n[Output]: ...{completion}")