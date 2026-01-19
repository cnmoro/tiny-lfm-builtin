import os
import tiny_lfm_builtin
from typing import List, Dict, Optional, Union, Generator

class TinyLFM:
    def __init__(self):
        """
        Initialize the Liquid LFM model.
        
        Args:
            model_path (str): Path to the GGUF model file.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "model-q4.gguf")
        self._engine = tiny_lfm_builtin.LiquidLFM(model_path)

    def chat(self,
             messages: List[Dict[str, str]], 
             max_tokens: int = None, 
             stream: bool = True) -> Union[str, Generator[str, None, None]]:
        """
        Regular chat generation. Maintains history automatically via the input list.
        KV Caching is handled automatically by the Rust engine based on prefix matching.

        Args:
            messages: List of dicts, e.g. [{"role": "user", "content": "..."}]
            max_tokens: Maximum new tokens to generate.
            stream: If True, returns a generator. If False, returns the full string.
        """
        streamer = self._engine.generate(messages, max_tokens) if max_tokens else self._engine.generate(messages)
        
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
        Allows pre-filling the assistant's response to guide output (e.g., forcing JSON).

        Args:
            prompt: The user's input/query.
            system_prompt: Optional system instruction.
            assistant_start: Text to pre-fill the assistant's response with. 
                             The model will continue generating from this point.
            stop: A string or list of strings that should stop generation.
            max_tokens: Max new tokens.
            stream: Yield tokens as they arrive.
        """
        # 1. Construct the raw prompt manually to allow template hacking
        full_prompt = ""
        
        if system_prompt:
            full_prompt += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            
        full_prompt += f"<|im_start|>user\n{prompt}<|im_end|>\n"
        full_prompt += "<|im_start|>assistant\n"
        
        if assistant_start:
            # We append the pre-fill without an EOS token, so the model continues it
            full_prompt += assistant_start

        # 2. Call Rust Engine
        # The engine will automatically check if 'full_prompt' shares a prefix 
        # with the previous generation and reuse the KV cache.
        streamer = self._engine.completion(full_prompt, max_tokens) if max_tokens else self._engine.completion(full_prompt)

        # 3. Handle Python-side stop tokens
        stop_sequences = []
        if stop:
            stop_sequences = [stop] if isinstance(stop, str) else stop
            
        generator = self._stop_aware_iterator(streamer, stop_sequences)

        if stream:
            return generator
        else:
            return "".join(list(generator))

    def save_cache(self, session_name: str):
        """Saves the current KV cache to disk."""
        self._engine.save_session(session_name)

    def load_cache(self, session_name: str):
        """Loads a KV cache from disk."""
        self._engine.load_session(session_name)

    def _stream_wrapper(self, rust_streamer) -> Generator[str, None, None]:
        """Simple wrapper to yield from Rust streamer."""
        for token in rust_streamer:
            yield token

    def _stop_aware_iterator(self, rust_streamer, stop_sequences: List[str]) -> Generator[str, None, None]:
        """
        Wraps the Rust streamer to implement custom stop sequences in Python.
        Note: The Rust engine handles standard EOS (<|im_end|>) internally.
        """
        generated_text = ""
        
        for token in rust_streamer:
            yield token
            generated_text += token
            
            # Check for stop sequences
            if stop_sequences:
                for seq in stop_sequences:
                    if seq in generated_text:
                        return # Stop generation immediately

if __name__ == "__main__":
    try:
        lfm = TinyLFM("model-q4.gguf")
        
        print("\n--- 1. Regular Chat Streaming ---")
        history = [{"role": "user", "content": "What is 2+2?"}]
        for token in lfm.chat(history):
            print(token, end="", flush=True)
        print("\n")

        print("--- 2. Prompt Hacking (JSON Mode) ---")
        # Scenario: We want to extract keywords as a JSON list.
        
        sys_p = "You are a data extraction tool. Output only JSON."
        user_p = "Extract keywords from: 'Liquid AI released LFM2, a powerful edge model.'"
        pre_fill = "Sure, here are the keywords in JSON format:\n```json\n[\n"
                
        stream = lfm.completion(
            prompt=user_p,
            system_prompt=sys_p,
            assistant_start=pre_fill,
            stop="]", # Stop when it tries to close the block
            stream=True
        )
        
        for token in stream:
            print(token, end="", flush=True)
        print("\n")
        
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")
