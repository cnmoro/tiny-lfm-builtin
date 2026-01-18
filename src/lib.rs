use pyo3::prelude::*;
use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;
use minijinja::Environment;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{Write, BufReader};
use std::sync::{Arc, Mutex, Once};

mod model;
use model::{Lfm2Model, TOKENIZER_BYTES};

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
}

#[pymethods]
impl LfmStreamer {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<String>> {
        if slf.current_step >= slf.max_tokens { return Ok(None); }

        let _pos = slf.tokens.len(); 
        let session_len = slf.session_tokens_ref.lock().unwrap().len();

        let input_ids = vec![*slf.tokens.last().unwrap()];
        
        let input_tensor = Tensor::new(input_ids.as_slice(), &slf.device)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Forward
        let logits = slf.model.forward(&input_tensor, session_len - 1)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let logits = logits.squeeze(0)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // --- PERFORMANCE TRICK 2: Zero-Copy Selection ---
        // Old way: logits.to_vec1() -> Vec<f32> (Slow, allocs memory)
        // New way: tensor.argmax() -> Scalar (Fast, stays in SIMD registers)
        
        let next_token_scalar = logits
            .argmax(0) // 0 is the dimension to reduce
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
            .to_scalar::<u32>()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let next_token = next_token_scalar;

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
    session_tokens: Arc<Mutex<Vec<u32>>>,
}

#[derive(Serialize)]
struct ChatMessage { role: String, content: String }

#[pymethods]
impl LiquidLFM {
    #[new]
    fn new(model_path: String) -> PyResult<Self> {
        // --- PERFORMANCE TRICK 1: Auto-set Threads ---
        INIT_RAYON.call_once(|| {
            let physical_cores = num_cpus::get_physical();
            // Rayon (used by Candle internally for parallel ops) defaults to logical cores.
            // On Ryzen, logical cores (hyperthreading) slows down AVX math.
            // We force it to physical cores.
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(physical_cores)
                .build_global();
            println!("Rust: Auto-configured Rayon to use {} physical cores.", physical_cores);
        });

        let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
        println!("Loading LFM2 from {} on {:?}...", model_path, device);

        let model = Lfm2Model::load(&model_path, &device)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let tokenizer = Tokenizer::from_bytes(TOKENIZER_BYTES)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let mut env = Environment::new();
        env.add_template("chat", CHAT_TEMPLATE).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        Ok(LiquidLFM { 
            model: Arc::new(model), 
            tokenizer, 
            device, 
            env,
            session_tokens: Arc::new(Mutex::new(Vec::new()))
        })
    }

    fn save_session(&self, path: String) -> PyResult<()> {
        let state_path = format!("{}.safetensors", path);
        self.model.save_state(&state_path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let meta_path = format!("{}.json", path);
        let json = serde_json::to_string(&*self.session_tokens.lock().unwrap()).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let mut file = File::create(meta_path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    fn load_session(&mut self, path: String) -> PyResult<()> {
        let state_path = format!("{}.safetensors", path);
        self.model.load_state(&state_path, &self.device).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let meta_path = format!("{}.json", path);
        let file = File::open(meta_path)?;
        let reader = BufReader::new(file);
        let tokens: Vec<u32> = serde_json::from_reader(reader).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        *self.session_tokens.lock().unwrap() = tokens;
        let internal_len = self.model.get_seq_len();
        let history_len = self.session_tokens.lock().unwrap().len();
        if internal_len != history_len {
             println!("Warning: Loaded state length ({}) does not match token history ({})", internal_len, history_len);
        } else {
             println!("Session loaded. Context size: {} tokens.", history_len);
        }
        Ok(())
    }

    #[pyo3(signature = (messages_py, max_new_tokens=None))]
    fn generate(&mut self, messages_py: Vec<HashMap<String, String>>, max_new_tokens: Option<usize>) -> PyResult<LfmStreamer> {
        let messages: Vec<ChatMessage> = messages_py.iter().map(|m| {
            ChatMessage { role: m.get("role").unwrap_or(&"user".to_string()).clone(), content: m.get("content").unwrap_or(&"".to_string()).clone() }
        }).collect();

        let tmpl = self.env.get_template("chat").map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let formatted_prompt = tmpl.render(minijinja::context! { messages => messages, bos_token => "<|im_start|>", add_generation_prompt => true }).unwrap();
        
        let encoding = self.tokenizer.encode(formatted_prompt, true).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let new_tokens = encoding.get_ids().to_vec();
        
        // Smart Caching Logic
        let mut history = self.session_tokens.lock().unwrap();
        let cached_len = history.len();
        let new_len = new_tokens.len();
        let mut prefix_match_len = 0;
        let min_len = std::cmp::min(cached_len, new_len);
        
        for i in 0..min_len {
            if history[i] == new_tokens[i] { prefix_match_len += 1; } else { break; }
        }

        if prefix_match_len == cached_len && cached_len > 0 {
             // Cache Hit
        } else {
            self.model.reset_internal_state();
            history.clear();
            prefix_match_len = 0;
        }

        if prefix_match_len < new_len {
            let tokens_to_feed = &new_tokens[prefix_match_len..];
            
            // OPTIMIZATION: Process in chunks to keep data in CPU Cache
            let chunk_size = 512; 
            
            // We feed everything EXCEPT the last token (handled by Streamer)
            if tokens_to_feed.len() > 1 {
                let bulk_feed = &tokens_to_feed[..tokens_to_feed.len() - 1];
                let mut current_pos = prefix_match_len;

                for chunk in bulk_feed.chunks(chunk_size) {
                    let input_tensor = Tensor::new(chunk, &self.device)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
                        .unsqueeze(0)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                    
                    self.model.forward(&input_tensor, current_pos)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                        
                    current_pos += chunk.len();
                }
            }
            
            *history = new_tokens.clone();
        }
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
        })
    }

    #[pyo3(signature = (prompt, max_new_tokens=None))]
    fn completion(&mut self, prompt: String, max_new_tokens: Option<usize>) -> PyResult<LfmStreamer> {
        // 1. Tokenize
        let encoding = self.tokenizer.encode(prompt, true)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let new_tokens = encoding.get_ids().to_vec();

        // 2. Manage Cache (Prefix Matching)
        let mut history = self.session_tokens.lock().unwrap();
        let cached_len = history.len();
        let new_len = new_tokens.len();
        
        let mut prefix_match_len = 0;
        let min_len = std::cmp::min(cached_len, new_len);
        for i in 0..min_len {
            if history[i] == new_tokens[i] { prefix_match_len += 1; } else { break; }
        }

        // If we are changing the past (diverged before the end of cache), we must full reset
        // because model.rs currently only supports full reset, not partial rollback.
        if prefix_match_len < cached_len {
             self.model.reset_internal_state();
             history.clear();
             prefix_match_len = 0;
        }

        // 3. Feed Tokens (Chunked Optimization)
        if prefix_match_len < new_len {
            let tokens_to_feed = &new_tokens[prefix_match_len..];
            
            // We feed N-1 tokens. The LAST token is fed by the Streamer's first step
            // to generate the first actual new token.
            if tokens_to_feed.len() > 1 {
                let bulk_feed = &tokens_to_feed[..tokens_to_feed.len() - 1];
                
                // Chunking to fit in CPU L3 Cache (Significant speedup for 4-bit)
                let chunk_size = 512;
                let mut current_pos = prefix_match_len;

                for chunk in bulk_feed.chunks(chunk_size) {
                    let input_tensor = Tensor::new(chunk, &self.device)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
                        .unsqueeze(0)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                    
                    self.model.forward(&input_tensor, current_pos)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

                    current_pos += chunk.len();
                }
            }

            *history = new_tokens.clone();
        }
        drop(history);

        // 4. Return Streamer
        Ok(LfmStreamer {
            model: self.model.clone(),
            tokenizer: self.tokenizer.clone(),
            device: self.device.clone(),
            tokens: new_tokens,
            max_tokens: max_new_tokens.unwrap_or(4096),
            current_step: 0,
            stop_tokens: HashSet::from([2]), // Standard EOS
            session_tokens_ref: self.session_tokens.clone(),
        })
    }
}

#[pymodule]
fn tiny_lfm_builtin(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LiquidLFM>()?;
    m.add_class::<LfmStreamer>()?;
    Ok(())
}