import os, sys, tiny_lfm_builtin
from typing import List, Dict, Optional, Union, Generator
import urllib.request
from enum import Enum

# Configuration
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "tiny_lfm_builtin")

class ModelType(str, Enum):
    M350 = "350M"
    B1_2 = "1.2B"
    B2_6 = "2.6B"
    B8_A1B = "8B-A1B"

# Model Registry
MODEL_REGISTRY = {
    ModelType.M350: {
        "url": "https://huggingface.co/cnmoro/LFM-Q4-GGUFS/resolve/main/lfm2-350m-q4.gguf",
        "filename": "lfm2-350m-q4.gguf"
    },
    ModelType.B1_2: {
        "url": "https://huggingface.co/cnmoro/LFM-Q4-GGUFS/resolve/main/lfm2.5-1.2b-q4.gguf",
        "filename": "lfm2.5-1.2b-q4.gguf"
    },
    ModelType.B2_6: {
        "url": "https://huggingface.co/cnmoro/LFM-Q4-GGUFS/resolve/main/lfm2-2.6b-q4.gguf",
        "filename": "lfm2-2.6b-q4.gguf"
    },
    ModelType.B8_A1B: {
        "url": "https://huggingface.co/cnmoro/LFM-Q4-GGUFS/resolve/main/lfm2-8b-a1b-q4.gguf",
        "filename": "lfm2-8b-a1b-q4.gguf"
    }
}

class TinyLFM:
    def __init__(self, model_size: Union[str, ModelType] = "350M", device: Optional[str] = None):
        """
        Initialize the Liquid LFM model.
        
        Args:
            model_size (str): One of "350M", "1.2B", "2.6B", "8B-A1B".
            device (str): "cpu", "cuda", or "metal". If None, auto-detects.
        """
        # Validate model selection
        if isinstance(model_size, str):
            try:
                model_size = ModelType(model_size)
            except ValueError:
                valid = [m.value for m in ModelType]
                raise ValueError(f"Invalid model_size '{model_size}'. Valid options: {valid}")
        
        config = MODEL_REGISTRY[model_size]
        self.model_filename = config["filename"]
        self.model_url = config["url"]
        
        os.makedirs(CACHE_DIR, exist_ok=True)
        model_path = os.path.join(CACHE_DIR, self.model_filename)
        
        if not os.path.exists(model_path):
            self._download_model(model_path)
        
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model file not found at: {model_path}")

        print(f"Loading LFM Engine ({model_size.value}) from {model_path}...")
        self._engine = tiny_lfm_builtin.LiquidLFM(model_path, device)
        print("Engine loaded.")

    def _download_model(self, dest_path: str):
        print(f"Model not found locally.")
        print(f"Downloading {self.model_filename} to {dest_path}...")
        
        def _progress(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\rDownload: {percent}%")
                sys.stdout.flush()

        try:
            urllib.request.urlretrieve(self.model_url, dest_path, reporthook=_progress)
            print("\nDownload complete.")
        except KeyboardInterrupt:
            print("\nDownload cancelled.")
            if os.path.exists(dest_path): os.remove(dest_path)
            sys.exit(1)
        except Exception as e:
            print(f"\nError downloading model: {e}")
            if os.path.exists(dest_path): os.remove(dest_path)
            raise e
    
    # --- Stateful Methods (Interactive Chat / Context Aware) ---

    def chat(self,
             messages: List[Dict[str, str]], 
             max_tokens: int = None, 
             stream: bool = True,
             ignore_thinking: bool = False) -> Union[str, Generator[str, None, None]]:
        """
        Standard interactive chat. 
        Maintains KV cache state based on the prefix of the conversation history.
        Best for single-user sessions or CLI chat bots.
        """
        _igthk = True if (ignore_thinking == False and "2.6b" in self.model_filename.lower()) else ignore_thinking
        streamer = self._engine.generate(messages, max_tokens, _igthk)
        
        if stream:
            return self._stream_wrapper(streamer)
        else:
            return "".join(list(streamer))

    def completion(self, 
                   prompt: str, 
                   system_prompt: Optional[str] = None, 
                   assistant_start: Optional[str] = None, 
                   stop: Optional[Union[str, List[str]]] = None,
                   max_tokens: int = None, 
                   stream: bool = True) -> Union[str, Generator[str, None, None]]:
        """
        Raw completion with 'Prompt Hacking' capabilities.
        Uses the stateful KV cache.
        """
        full_prompt = ""
        if system_prompt:
            full_prompt += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        full_prompt += f"<|im_start|>user\n{prompt}<|im_end|>\n"
        full_prompt += "<|im_start|>assistant\n"
        
        # If model is 2.6B then add <think>\n</think> to assistant msg prefix before anything else
        # (This is manual handling for completion since completion bypasses 'generate')
        if "2.6b" in self.model_filename.lower():
            full_prompt += "<think>\n</think>\n"

        if assistant_start:
            full_prompt += assistant_start

        streamer = self._engine.completion(full_prompt, max_tokens)

        stop_sequences = [stop] if isinstance(stop, str) else (stop or [])
        generator = self._stop_aware_iterator(streamer, stop_sequences)

        if stream:
            return generator
        else:
            return "".join(list(generator))

    def clear_session(self):
        """
        Manually clears the stateful KV cache and session history to free RAM.
        Use this when starting a completely unrelated conversation in 'chat' or 'completion' mode.
        """
        self._engine.clear_session()

    def save_cache(self, session_name: str):
        """Saves the current stateful KV cache to disk."""
        self._engine.save_session(session_name)

    def load_cache(self, session_name: str):
        """Loads a stateful KV cache from disk."""
        self._engine.load_session(session_name)

    # --- Stateless / High-Throughput Methods ---

    def chat_stateless(self, 
                       messages: List[Dict[str, str]], 
                       max_tokens: int = None,
                       ignore_thinking: bool = False) -> str:
        """
        Thread-safe, stateless chat. 
        Releases the Python GIL during inference.
        
        Use this method when running a web server (e.g., FastAPI) to handle 
        multiple concurrent requests in separate threads without blocking.
        
        Returns:
            str: The full response string (streaming not supported in stateless mode yet).
        """
        _igthk = True if (ignore_thinking == False and "2.6b" in self.model_filename.lower()) else ignore_thinking
        return self._engine.chat_stateless(messages, max_tokens, _igthk)

    def batch_chat(self, 
                   conversations: List[List[Dict[str, str]]], 
                   max_tokens: int = None,
                   ignore_thinking: bool = False) -> List[str]:
        """
        Process multiple chat conversations in parallel using CPU multithreading (Rayon).
        Each conversation runs in isolation with its own temporary KV cache.
        
        Args:
            conversations: A list of chat histories (list of lists of dicts).
        
        Returns:
            List[str]: A list of full text responses (including the prompt).
        """
        _igthk = True if (ignore_thinking == False and "2.6b" in self.model_filename.lower()) else ignore_thinking
        return self._engine.batch_chat(conversations, max_tokens, _igthk)

    def batch_completion(self, 
                         prompts: List[str], 
                         max_tokens: int = None) -> List[str]:
        """
        Process multiple raw completions in parallel using CPU multithreading (Rayon).
        
        Args:
            prompts: A list of string prompts.
            
        Returns:
            List[str]: A list of full text responses (including the prompt).
        """
        return self._engine.batch_completion(prompts, max_tokens)

    # --- Helpers ---

    def _stream_wrapper(self, rust_streamer) -> Generator[str, None, None]:
        for token in rust_streamer:
            yield token

    def _stop_aware_iterator(self, rust_streamer, stop_sequences: List[str]) -> Generator[str, None, None]:
        generated_text = ""
        for token in rust_streamer:
            yield token
            generated_text += token
            if stop_sequences:
                for seq in stop_sequences:
                    if seq in generated_text:
                        return

if __name__ == "__main__":
    try:
        # Example Usage
        print("Initializing 2.6B Model...")
        lfm = TinyLFM("1.2B")
        
        print("\n1. Testing standard chat...")
        for t in lfm.chat([{"role": "user", "content": "Hi!"}]):
            print(t, end="", flush=True)
        print("\n")
        
        print("2. Testing Batch Chat...")
        batch_inputs = [
            # [{"role": "user", "content": "1+1="}],
            [{"role": "user", "content": "Capital of Spain?"}]
        ]
        results = lfm.batch_chat(batch_inputs, max_tokens=80)
        for r in results:
            print(f"Result: {r[-50:].strip()}...")

        print("\n3. Testing Resource Cleanup...")
        lfm.clear_session()
        print("Session cleared.")

    except Exception as e:
        print(f"Error: {e}")