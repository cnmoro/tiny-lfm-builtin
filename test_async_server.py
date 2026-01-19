from tiny_lfm import TinyLFM
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Initialize model
model = TinyLFM()

# Simulate a thread pool (like FastAPI uses)
executor = ThreadPoolExecutor(max_workers=4)

def handle_request(req_id, prompt, delay):
    """
    Simulates a request handler.
    1. Waits a bit (simulating different arrival time).
    2. Calls the blocking Rust model (but Rust releases GIL).
    """
    time.sleep(delay)
    print(f"[{req_id}] Started processing: '{prompt[:20]}...'")
    
    chat_input = [{"role": "user", "content": prompt}]
    
    # This call releases GIL in Rust, so other threads can run!
    result = model.chat_stateless(chat_input, max_tokens=40)
    
    print(f"[{req_id}] Finished.")
    return result

async def main():
    print("--- Starting Async Server Simulation ---")
    
    loop = asyncio.get_running_loop()
    
    # We simulate 3 requests arriving at different times:
    # Req 1: Immediate
    # Req 2: 1 second later
    # Req 3: 2 seconds later
    
    prompts = [
        (1, "What is the capital of France?", 0),
        (2, "Explain Quantum Physics briefly.", 1),
        (3, "Write a haiku about rust.", 2)
    ]
    
    tasks = []
    start_total = time.time()

    for req_id, prompt, delay in prompts:
        # run_in_executor runs the blocking function in a separate thread
        task = loop.run_in_executor(executor, handle_request, req_id, prompt, delay)
        tasks.append(task)
        
    results = await asyncio.gather(*tasks)
    
    print(f"--- All processed in {time.time() - start_total:.2f}s ---")
    
    for r in results:
        print(f"Response: {r[-50:].strip()}...") # Print tail

if __name__ == "__main__":
    asyncio.run(main())