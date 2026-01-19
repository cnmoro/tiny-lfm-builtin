use pyo3::prelude::*;
use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;
use minijinja::Environment;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{Write, BufReader};
use std::sync::{Arc, Mutex, Once};
use rayon::prelude::*;

mod model;
use model::{Lfm2Model, LfmCache, TOKENIZER_BYTES};

const CHAT_TEMPLATE: &str = include_str!("../chat_template.jinja");
static INIT_RAYON: Once = Once::new();

#[pyclass]
struct LfmStreamer {
    model: Arc<Lfm2Model>,
    tokenizer: Tokenizer,
    device: Device,
    tokens: Vec<u32>,
    max_tokens: usize,
    current_step: usize,
    stop_tokens: HashSet<u32>, 
    session_tokens_ref: Arc<Mutex<Vec<u32>>>,
    cache: Arc<Mutex<LfmCache>>,
}

#[pymethods]
impl LfmStreamer {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<String>> {
        if slf.current_step >= slf.max_tokens { return Ok(None); }

        let session_len = slf.session_tokens_ref.lock().unwrap().len();
        let input_ids = vec![*slf.tokens.last().unwrap()];
        
        let input_tensor = Tensor::new(input_ids.as_slice(), &slf.device)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let cache_ref = slf.cache.clone();
        let mut cache = cache_ref.lock().unwrap();

        let logits = slf.model.forward(&input_tensor, session_len - 1, &mut *cache)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let logits = logits.squeeze(0)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let next_token = logits.argmax(0)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
            .to_scalar::<u32>()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        if slf.stop_tokens.contains(&next_token) { return Ok(None); }

        slf.tokens.push(next_token);
        slf.session_tokens_ref.lock().unwrap().push(next_token);
        slf.current_step += 1;

        let text = slf.tokenizer.decode(&[next_token], true).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Some(text))
    }
}

#[pyclass]
struct LiquidLFM {
    model: Arc<Lfm2Model>,
    tokenizer: Tokenizer,
    device: Device,
    env: Environment<'static>,
    // Stateful session storage
    session_tokens: Arc<Mutex<Vec<u32>>>,
    cache: Arc<Mutex<LfmCache>>,
}

#[derive(Serialize)]
struct ChatMessage { role: String, content: String }

impl LiquidLFM {
    // Helper to run inference on a batch of token vectors using Rayon
    fn run_parallel_inference(&self, inputs: Vec<Vec<u32>>, max_new_tokens: usize) -> Result<Vec<String>, candle_core::Error> {
        let model = &self.model;
        let device = &self.device;
        let tokenizer = &self.tokenizer;
        let num_layers = model.num_layers();

        inputs.into_par_iter().map(|mut tokens| {
            let mut local_cache = LfmCache::new(num_layers);
            let mut current_pos = 0;

            // Prefill: Process all tokens except the last one
            let prefill_len = tokens.len().saturating_sub(1);
            if prefill_len > 0 {
                for chunk in tokens[..prefill_len].chunks(512) {
                    let input_tensor = Tensor::new(chunk, device)?.unsqueeze(0)?;
                    model.forward(&input_tensor, current_pos, &mut local_cache)?;
                    current_pos += chunk.len();
                }
            }

            // Generation
            for _ in 0..max_new_tokens {
                let last_token = *tokens.last().unwrap();
                let input = Tensor::new(&[last_token], device)?.unsqueeze(0)?;
                
                let logits = model.forward(&input, current_pos, &mut local_cache)?;
                let logits = logits.squeeze(0)?.squeeze(0)?;
                
                let next_token = logits.argmax(0)?.to_scalar::<u32>()?;
                tokens.push(next_token);
                current_pos += 1;

                if next_token == 2 || next_token == 32000 { break; } 
            }

            tokenizer.decode(&tokens, true).map_err(|e| candle_core::Error::Msg(e.to_string()))
        }).collect()
    }
}

#[pymethods]
impl LiquidLFM {
    #[new]
    #[pyo3(signature = (model_path, device=None))]
    fn new(model_path: String, device: Option<String>) -> PyResult<Self> {
        INIT_RAYON.call_once(|| {
            let physical_cores = num_cpus::get_physical();
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(physical_cores)
                .build_global();
        });

        let device = match device.as_deref() {
            Some("cpu") => {
                println!("Rust: User requested CPU.");
                Device::Cpu
            },
            Some("cuda") => {
                 if candle_core::utils::cuda_is_available() {
                    println!("Rust: User requested CUDA.");
                    Device::new_cuda(0).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
                 } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("CUDA requested but not available."));
                 }
            },
            Some("metal") => {
                 if candle_core::utils::metal_is_available() {
                    println!("Rust: User requested Metal.");
                    Device::new_metal(0).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
                 } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Metal requested but not available."));
                 }
            },
            _ => {
                // Auto-detection
                if candle_core::utils::cuda_is_available() {
                    println!("Rust: CUDA detected. Using GPU.");
                    Device::new_cuda(0).unwrap_or(Device::Cpu)
                } else if candle_core::utils::metal_is_available() {
                    println!("Rust: Metal detected. Using GPU.");
                    Device::new_metal(0).unwrap_or(Device::Cpu)
                } else {
                    println!("Rust: Using CPU.");
                    Device::Cpu
                }
            }
        };

        let model = Lfm2Model::load(&model_path, &device)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let tokenizer = Tokenizer::from_bytes(TOKENIZER_BYTES)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        
        let mut env = Environment::new();
        env.add_template("chat", CHAT_TEMPLATE).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        let num_layers = model.num_layers();

        Ok(LiquidLFM { 
            model: Arc::new(model), 
            tokenizer, 
            device, 
            env,
            session_tokens: Arc::new(Mutex::new(Vec::new())),
            cache: Arc::new(Mutex::new(LfmCache::new(num_layers)))
        })
    }

    fn clear_session(&self) -> PyResult<()> {
        let mut cache = self.cache.lock().unwrap();
        let mut tokens = self.session_tokens.lock().unwrap();
        *cache = LfmCache::new(self.model.num_layers());
        tokens.clear();
        Ok(())
    }

    fn save_session(&self, path: String) -> PyResult<()> {
        let state_path = format!("{}.safetensors", path);
        self.cache.lock().unwrap().save(&state_path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        
        let meta_path = format!("{}.json", path);
        let json = serde_json::to_string(&*self.session_tokens.lock().unwrap()).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let mut file = File::create(meta_path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    fn load_session(&mut self, path: String) -> PyResult<()> {
        let state_path = format!("{}.safetensors", path);
        self.cache.lock().unwrap().load(&state_path, &self.device).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        
        let meta_path = format!("{}.json", path);
        let file = File::open(meta_path)?;
        let reader = BufReader::new(file);
        let tokens: Vec<u32> = serde_json::from_reader(reader).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        *self.session_tokens.lock().unwrap() = tokens;
        
        let internal_len = self.cache.lock().unwrap().get_seq_len();
        let history_len = self.session_tokens.lock().unwrap().len();
        if internal_len != history_len {
             println!("Warning: Loaded state length ({}) does not match token history ({})", internal_len, history_len);
        }
        Ok(())
    }

    // --- BATCH API (Rayon) ---

    #[pyo3(signature = (prompts, max_new_tokens=None))]
    fn batch_completion(&self, prompts: Vec<String>, max_new_tokens: Option<usize>) -> PyResult<Vec<String>> {
        let max_tokens = max_new_tokens.unwrap_or(64);
        
        let inputs: Result<Vec<Vec<u32>>, _> = prompts.iter().map(|p| {
            // Keep true for raw completion as user might expect BOS
            self.tokenizer.encode(p.clone(), true)
                .map(|e| e.get_ids().to_vec())
                .map_err(|e| candle_core::Error::Msg(e.to_string()))
        }).collect();
        
        let inputs = inputs.map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        self.run_parallel_inference(inputs, max_tokens)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    #[pyo3(signature = (conversations, max_new_tokens=None, ignore_thinking=false))]
    fn batch_chat(&self, conversations: Vec<Vec<HashMap<String, String>>>, max_new_tokens: Option<usize>, ignore_thinking: bool) -> PyResult<Vec<String>> {
        let max_tokens = max_new_tokens.unwrap_or(64);

        let tmpl = self.env.get_template("chat")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        
        let mut inputs = Vec::new();
        for messages_py in conversations {
            let messages: Vec<ChatMessage> = messages_py.iter().map(|m| {
                ChatMessage { 
                    role: m.get("role").unwrap_or(&"user".to_string()).clone(), 
                    content: m.get("content").unwrap_or(&"".to_string()).clone() 
                }
            }).collect();

            let mut formatted_prompt = tmpl.render(minijinja::context! { 
                messages => messages, 
                bos_token => "", 
                add_generation_prompt => true 
            }).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

            if ignore_thinking {
                formatted_prompt.push_str("<think>\n</think>\n");
            }

            // print the raw prompt
            println!("Raw prompt:\n{}", formatted_prompt);

            // Use `true` to ensure the BOS token ID (<|startoftext|>) is added automatically.
            let encoding = self.tokenizer.encode(formatted_prompt, true)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            
            inputs.push(encoding.get_ids().to_vec());
        }

        self.run_parallel_inference(inputs, max_tokens)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    // --- ASYNC/STATELESS API (GIL Released) ---

    #[pyo3(signature = (messages_py, max_new_tokens=None, ignore_thinking=false))]
    fn chat_stateless(&self, py: Python<'_>, messages_py: Vec<HashMap<String, String>>, max_new_tokens: Option<usize>, ignore_thinking: bool) -> PyResult<String> {
        let max_tokens = max_new_tokens.unwrap_or(64);
        
        // 1. Prepare Data (Holding GIL)
        let messages: Vec<ChatMessage> = messages_py.iter().map(|m| {
            ChatMessage { 
                role: m.get("role").unwrap_or(&"user".to_string()).clone(), 
                content: m.get("content").unwrap_or(&"".to_string()).clone() 
            }
        }).collect();

        let tmpl = self.env.get_template("chat").map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        
        let mut formatted_prompt = tmpl.render(minijinja::context! { 
            messages => messages, 
            bos_token => "", 
            add_generation_prompt => true 
        }).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        if ignore_thinking {
            formatted_prompt.push_str("<think>\n</think>\n");
        }

        // print the raw prompt
        println!("Raw prompt:\n{}", formatted_prompt);

        // Use `true` to ensure the BOS token ID (<|startoftext|>) is added automatically.
        let encoding = self.tokenizer.encode(formatted_prompt, true).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let tokens = encoding.get_ids().to_vec();

        // 2. Run Inference (Releasing GIL)
        let model_ref = self.model.clone();
        let tokenizer_ref = self.tokenizer.clone();
        let device_ref = self.device.clone();

        let result: Result<String, candle_core::Error> = py.allow_threads(move || {
            let num_layers = model_ref.num_layers();
            let mut local_cache = LfmCache::new(num_layers);
            let mut current_pos = 0;
            let mut local_tokens = tokens.clone();

            // Prefill: Process all tokens except the last one
            let prefill_len = local_tokens.len().saturating_sub(1);
            if prefill_len > 0 {
                for chunk in local_tokens[..prefill_len].chunks(512) {
                    let input = Tensor::new(chunk, &device_ref)?.unsqueeze(0)?;
                    model_ref.forward(&input, current_pos, &mut local_cache)?;
                    current_pos += chunk.len();
                }
            }

            for _ in 0..max_tokens {
                let last = *local_tokens.last().unwrap();
                let input = Tensor::new(&[last], &device_ref)?.unsqueeze(0)?;
                
                // Process the last token (or subsequently generated tokens)
                let logits = model_ref.forward(&input, current_pos, &mut local_cache)?.squeeze(0)?.squeeze(0)?;
                let next = logits.argmax(0)?.to_scalar::<u32>()?;
                
                local_tokens.push(next);
                current_pos += 1;
                if next == 2 || next == 32000 { break; }
            }
            tokenizer_ref.decode(&local_tokens, true).map_err(|e| candle_core::Error::Msg(e.to_string()))
        });

        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    // --- STATEFUL/STREAMING API (Single Thread) ---

    #[pyo3(signature = (messages_py, max_new_tokens=None, ignore_thinking=false))]
    fn generate(&mut self, messages_py: Vec<HashMap<String, String>>, max_new_tokens: Option<usize>, ignore_thinking: bool) -> PyResult<LfmStreamer> {
        let messages: Vec<ChatMessage> = messages_py.iter().map(|m| {
            ChatMessage { role: m.get("role").unwrap_or(&"user".to_string()).clone(), content: m.get("content").unwrap_or(&"".to_string()).clone() }
        }).collect();

        let tmpl = self.env.get_template("chat").map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        
        let mut formatted_prompt = tmpl.render(minijinja::context! { 
            messages => messages, 
            bos_token => "", 
            add_generation_prompt => true 
        }).unwrap();
        
        if ignore_thinking {
            formatted_prompt.push_str("<think>\n</think>\n");
        }

        // Use `true` to ensure the BOS token ID (<|startoftext|>) is added automatically.
        let encoding = self.tokenizer.encode(formatted_prompt, true).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let new_tokens = encoding.get_ids().to_vec();
        
        let mut history = self.session_tokens.lock().unwrap();
        let cached_len = history.len();
        let new_len = new_tokens.len();
        let mut prefix_match_len = 0;
        let min_len = std::cmp::min(cached_len, new_len);
        
        for i in 0..min_len {
            if history[i] == new_tokens[i] { prefix_match_len += 1; } else { break; }
        }

        let mut cache = self.cache.lock().unwrap();

        if prefix_match_len == cached_len && cached_len > 0 {
             // Cache Hit
        } else {
            *cache = LfmCache::new(self.model.num_layers());
            history.clear();
            prefix_match_len = 0;
        }

        if prefix_match_len < new_len {
            let tokens_to_feed = &new_tokens[prefix_match_len..];
            let chunk_size = 512; 
            
            if tokens_to_feed.len() > 1 {
                let bulk_feed = &tokens_to_feed[..tokens_to_feed.len() - 1];
                let mut current_pos = prefix_match_len;

                for chunk in bulk_feed.chunks(chunk_size) {
                    let input_tensor = Tensor::new(chunk, &self.device)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
                        .unsqueeze(0)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                    
                    self.model.forward(&input_tensor, current_pos, &mut *cache)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                        
                    current_pos += chunk.len();
                }
            }
            *history = new_tokens.clone();
        }
        drop(cache);
        drop(history);

        let mut stop_tokens = HashSet::new();
        stop_tokens.insert(2); 
        for s in ["<|im_end|>", "<|tool_response_end|>"] { if let Some(id) = self.tokenizer.token_to_id(s) { stop_tokens.insert(id); } }

        Ok(LfmStreamer {
            model: self.model.clone(),
            tokenizer: self.tokenizer.clone(),
            device: self.device.clone(),
            tokens: new_tokens,
            max_tokens: max_new_tokens.unwrap_or(4096),
            current_step: 0,
            stop_tokens,
            session_tokens_ref: self.session_tokens.clone(),
            cache: self.cache.clone(),
        })
    }

    #[pyo3(signature = (prompt, max_new_tokens=None))]
    fn completion(&mut self, prompt: String, max_new_tokens: Option<usize>) -> PyResult<LfmStreamer> {
        let encoding = self.tokenizer.encode(prompt, true)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let new_tokens = encoding.get_ids().to_vec();

        let mut history = self.session_tokens.lock().unwrap();
        let cached_len = history.len();
        let new_len = new_tokens.len();
        
        let mut prefix_match_len = 0;
        let min_len = std::cmp::min(cached_len, new_len);
        for i in 0..min_len {
            if history[i] == new_tokens[i] { prefix_match_len += 1; } else { break; }
        }

        let mut cache = self.cache.lock().unwrap();

        if prefix_match_len < cached_len {
             *cache = LfmCache::new(self.model.num_layers());
             history.clear();
             prefix_match_len = 0;
        }

        if prefix_match_len < new_len {
            let tokens_to_feed = &new_tokens[prefix_match_len..];
            
            if tokens_to_feed.len() > 1 {
                let bulk_feed = &tokens_to_feed[..tokens_to_feed.len() - 1];
                let chunk_size = 512;
                let mut current_pos = prefix_match_len;

                for chunk in bulk_feed.chunks(chunk_size) {
                    let input_tensor = Tensor::new(chunk, &self.device)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
                        .unsqueeze(0)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                    
                    self.model.forward(&input_tensor, current_pos, &mut *cache)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

                    current_pos += chunk.len();
                }
            }

            *history = new_tokens.clone();
        }
        drop(cache);
        drop(history);

        Ok(LfmStreamer {
            model: self.model.clone(),
            tokenizer: self.tokenizer.clone(),
            device: self.device.clone(),
            tokens: new_tokens,
            max_tokens: max_new_tokens.unwrap_or(4096),
            current_step: 0,
            stop_tokens: HashSet::from([2]), 
            session_tokens_ref: self.session_tokens.clone(),
            cache: self.cache.clone(),
        })
    }
}

#[pymodule]
fn tiny_lfm_builtin(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LiquidLFM>()?;
    m.add_class::<LfmStreamer>()?;
    Ok(())
}