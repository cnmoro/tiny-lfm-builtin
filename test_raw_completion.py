import lfm_rust

model = lfm_rust.LiquidLFM("model-q4.gguf")

# 1. Test Completion (1 token)
print("--- Completion Test ---")
stream = model.completion("The opposite of hot is ", max_new_tokens=1) # cold
for t in stream: print(t, end="")
print("\n")
