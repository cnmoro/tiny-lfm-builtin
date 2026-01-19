import tiny_lfm_builtin
import time

model = tiny_lfm_builtin.LiquidLFM("model-q4.gguf")

history = [{"role": "user", "content": """
Context:
            
"Cleaning the toilet is a task that doesn't interest people. Many, however, pray for technology that can save them from the unpleasant mission. Apparently, those prayers were answered: a group of Chinese scientists developed the concept of a self-cleaning toilet and managed to make it a reality. Thanks to 3D printing, researchers at Huazhong University of Science and Technology have managed to revolutionize the unpleasant household chore. The self-cleaning toilet, known as “ARSFT”, an acronym for “abrasion-resistant super slippery toilet flush” — the technology that allows automatic cleaning — emerged from a complex combination of plastic and grains of sand that repel water. In plain English, the technology ensures that no substance sticks to the surface. Therefore, in addition to being a salvation for many, this can be a more sustainable alternative to conventional toilets. The website New Scientist interviewed one of the project's scientists, Yike Li, who created the self-cleaning toilet. According to Li, the Chinese used, in addition to the combination of plastic and grains of sand, a laser to bring the particles together, thus creating the 3D printed self-cleaning toilet. After printing, the researchers used silicon oil to lubricate the surface of the toilet, managing to penetrate it due to the structure of the model. This generated the toilet's self-cleaning capacity, with the following materials leaving no marks after flushing: Milk; Yogurt; Honey; Muddy water; Starch gel mixed with porridge. Chinese scientists also tested the self-cleaning toilet with synthetic feces, using a mixture of miso, yeast, peanut oil and water, managing to imitate human excrement. Although it may be strange that scientists work to create toilet technologies, several seemingly “unnecessary” innovations can have a major global impact. The self-cleaning toilet created by Chinese researchers can considerably reduce water waste. According to Chinese scientists, the self-cleaning toilet can withstand a thousand scraping cycles thanks to its super slippery capacity. Therefore, the self-cleaning toilet has a new flushing method that minimizes water consumption – and waste. The Daily Mail points out that, since its invention in the 18th century, although the toilet has increased hygiene, a significant amount of water is required due to the adhesion between the surface of the toilet and human feces and urine. Worldwide, toilet flushes correspond to 141 billion liters of water daily. Therefore, in addition to saving a valuable resource for humanity, the self-cleaning toilet also has another environmental benefit. In places such as public and chemical bathrooms, especially where there is no connection to the sanitation system, the self-cleaning toilet appears as an ideal solution."

Read the context carefully""".strip()}]

print("--- Turn 1 (Cold Start) ---")
t0 = time.time()
stream = model.generate(history)
for t in stream: pass 
print(f"Time taken: {time.time() - t0:.2f}s")

print("--- Saving Cache ---")
# This saves 'my_chat.safetensors' (KV) and 'my_chat.json' (Tokens)
model.save_session("my_chat")

# Simulate restarting the application
print("--- Reloading Model & Cache ---")
model2 = tiny_lfm_builtin.LiquidLFM("model-q4.gguf")
model2.load_session("my_chat")

print("--- Turn 2 (Warm Start) ---")
history.append({"role": "assistant", "content": "I've read the context carefully."})
history.append({"role": "user", "content": "Summarize the content in a single paragraph."})

t0 = time.time()
# The Rust logic will see that the start of the history matches the loaded cache.
stream = model2.generate(history)
for t in stream: print(t, end="", flush=True)
print(f"\nTime taken: {time.time() - t0:.2f}s")