from tiny_lfm import TinyLFM

model = TinyLFM()

# 1. Test Completion (1 token)
print("--- Completion Test ---")
stream = model.completion("The opposite of hot is ", max_tokens=1) # cold
for t in stream: print(t, end="")
print("\n")
