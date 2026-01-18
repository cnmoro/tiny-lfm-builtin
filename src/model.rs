use candle_core::quantized::{gguf_file, QMatMul, QTensor};
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Embedding, Linear};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::fs::File;
use std::io::{Read, Seek};

pub const TOKENIZER_BYTES: &[u8] = include_bytes!("../tokenizer.json");

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub norm_eps: f64,
    pub conv_l_cache: usize, 
    pub layer_types: Vec<String>,
    pub rope_theta: f32,
}

impl Config {
    pub fn from_gguf(content: &gguf_file::Content) -> Result<Self> {
        let get_val = |keys: &[&str]| -> Result<&gguf_file::Value> {
            for &k in keys {
                if let Some(v) = content.metadata.get(k) { return Ok(v); }
            }
            let available: Vec<_> = content.metadata.keys().take(10).collect();
            candle_core::bail!("Metadata key not found {:?}. Available: {:?}", keys, available)
        };

        let get_usize = |keys: &[&str]| -> Result<usize> {
            let v = get_val(keys)?;
            v.to_u32().map(|v| v as usize).or_else(|_| v.to_u64().map(|v| v as usize)).map_err(|_| candle_core::Error::Msg("Not int".into()))
        };
        let get_f64 = |keys: &[&str]| -> Result<f64> {
            let v = get_val(keys)?;
            v.to_f64().or_else(|_| v.to_f32().map(|v| v as f64)).map_err(|_| candle_core::Error::Msg("Not float".into()))
        };

        let hidden_size = get_usize(&["lfm2.embedding_length", "llama.embedding_length", "hidden_size"])?;
        let num_attention_heads = get_usize(&["lfm2.attention.head_count", "llama.attention.head_count", "num_attention_heads"])?;
        let num_hidden_layers = get_usize(&["lfm2.block_count", "llama.block_count", "num_hidden_layers"])?;
        // Default to attention_heads if KV missing (common cause of error, fixed in Attention::new via auto-detect)
        let num_key_value_heads = get_usize(&["lfm2.attention.head_count_kv", "llama.attention.head_count_kv", "num_key_value_heads"]).unwrap_or(num_attention_heads);
        let norm_eps = get_f64(&["lfm2.attention.layer_norm_rms_epsilon", "llama.attention.layer_norm_rms_epsilon", "norm_eps"])?;
        let rope_theta = get_f64(&["lfm2.rope.freq_base", "llama.rope.freq_base", "rope_theta"]).unwrap_or(10000.0);
        
        let conv_l_cache = get_usize(&["lfm2.conv_l_cache", "conv_L_cache"]).unwrap_or(4);

        let layer_types = match get_val(&["lfm2.layer_types", "layer_types"]) {
            Ok(v) => v.to_string()?.split(',').map(|s| s.to_string()).collect(),
            Err(_) => vec![]
        };

        Ok(Config {
            hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads,
            head_dim: hidden_size / num_attention_heads, norm_eps, conv_l_cache, layer_types, rope_theta: rope_theta as f32,
        })
    }
}

// Struct to hold Linear layer + its output dimension (crucial for weight shape checking)
struct QLinear {
    inner: QLinearInner,
    out_dim: usize,
}
enum QLinearInner {
    Standard(Linear),
    Quantized(QMatMul),
}

impl QLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.inner { 
            QLinearInner::Standard(l) => l.forward(x), 
            QLinearInner::Quantized(q) => q.forward(x) 
        }
    }
}

pub struct Weights {
    tensors: HashMap<String, QTensor>,
    device: Device,
}

impl Weights {
    pub fn load<R: Read + Seek>(reader: &mut R, content: &gguf_file::Content, device: &Device) -> Result<Self> {
        let mut tensors = HashMap::new();
        for (name, _) in &content.tensor_infos {
            tensors.insert(name.to_string(), content.tensor(reader, name, device)?);
        }
        Ok(Self { tensors, device: device.clone() })
    }

    pub fn pop_tensor(&mut self, name: &str) -> Result<Tensor> {
        if let Some(qt) = self.tensors.remove(name) { return qt.dequantize(&self.device); }
        let aliases = match name {
            "model.embed_tokens.weight" => vec!["token_embd.weight"],
            "model.embedding_norm.weight" => vec!["output_norm.weight", "token_embd_norm.weight", "ln_f.weight"],
            _ => vec![]
        };
        for alias in aliases {
            if let Some(qt) = self.tensors.remove(alias) { return qt.dequantize(&self.device); }
        }
        let keys: Vec<_> = self.tensors.keys().take(15).collect();
        candle_core::bail!("Missing tensor '{}'. First 15 keys: {:?}", name, keys)
    }

    pub fn pop_linear(&mut self, name: &str) -> Result<QLinear> {
        let w_qt = self.tensors.remove(name).ok_or_else(|| {
            let keys: Vec<_> = self.tensors.keys().take(10).collect();
            candle_core::Error::Msg(format!("Missing linear '{}'. Available: {:?}", name, keys))
        })?;

        let out_dim = w_qt.shape().dims()[0]; // Capture output dimension

        let inner = match w_qt.dtype() {
            candle_core::quantized::GgmlDType::F32 | candle_core::quantized::GgmlDType::F16 => {
                QLinearInner::Standard(Linear::new(w_qt.dequantize(&self.device)?, None))
            },
            _ => QLinearInner::Quantized(QMatMul::from_qtensor(w_qt)?)
        };
        Ok(QLinear { inner, out_dim })
    }
    
    pub fn has(&self, name: &str) -> bool { self.tensors.contains_key(name) }
}

struct RotaryEmbedding { cos: Tensor, sin: Tensor }
impl RotaryEmbedding {
    fn new(theta: f32, head_dim: usize, max_seq_len: usize, device: &Device) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..head_dim / 2).map(|i| 1.0 / theta.powf((2 * i) as f32 / head_dim as f32)).collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, device)?.to_dtype(DType::F32)?.reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self { cos: freqs.cos()?, sin: freqs.sin()? })
    }
    fn apply(&self, x: &Tensor, pos: usize) -> Result<Tensor> {
        let (_b, _h, seq_len, head_dim) = x.dims4()?;
        let cos = self.cos.narrow(0, pos, seq_len)?.reshape((1, 1, seq_len, head_dim / 2))?.to_dtype(x.dtype())?;
        let sin = self.sin.narrow(0, pos, seq_len)?.reshape((1, 1, seq_len, head_dim / 2))?.to_dtype(x.dtype())?;
        let x1 = x.narrow(3, 0, head_dim / 2)?;
        let x2 = x.narrow(3, head_dim / 2, head_dim / 2)?;
        let r1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let r2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
        Tensor::cat(&[r1, r2], 3)
    }
}

struct RmsNorm { weight: Tensor, eps: f64 }
impl RmsNorm {
    fn new(w: Tensor, eps: f64) -> Self { Self { weight: w, eps } }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let denom = (variance + self.eps)?.sqrt()?;
        let hidden = x_f32.broadcast_div(&denom)?.to_dtype(x_dtype)?;
        hidden.broadcast_mul(&self.weight)
    }
}

struct Mlp { w1: QLinear, w2: QLinear, w3: QLinear }
impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x1 = self.w1.forward(x)?;
        let x3 = self.w3.forward(x)?;
        let hidden = (candle_nn::ops::silu(&x1)? * x3)?;
        self.w2.forward(&hidden)
    }
}

struct ShortConv {
    conv_weight: Tensor, 
    conv_bias: Option<Tensor>,
    in_proj: QLinear, out_proj: QLinear, 
    l_cache: usize, cache: Arc<Mutex<Option<Tensor>>>, 
}
impl ShortConv {
    fn new(cfg: &Config, weights: &mut Weights, prefix: &str, gguf: bool) -> Result<Self> {
        let (w, b, in_n, out_n) = if gguf {
            (format!("{}.shortconv.conv.weight", prefix), format!("{}.shortconv.conv.bias", prefix), format!("{}.shortconv.in_proj.weight", prefix), format!("{}.shortconv.out_proj.weight", prefix))
        } else {
            (format!("{}.conv.weight", prefix), format!("{}.conv.bias", prefix), format!("{}.conv.in_proj.weight", prefix), format!("{}.conv.out_proj.weight", prefix))
        };
        
        let mut conv_w = weights.pop_tensor(&w)?;
        if conv_w.rank() == 2 { conv_w = conv_w.unsqueeze(1)?; }
        let l_cache = conv_w.dim(2).unwrap_or(cfg.conv_l_cache); // Auto-detect Kernel Size
        let conv_bias = if weights.has(&b) { Some(weights.pop_tensor(&b)?) } else { None };

        Ok(Self {
            conv_weight: conv_w, conv_bias,
            in_proj: weights.pop_linear(&in_n)?,
            out_proj: weights.pop_linear(&out_n)?,
            l_cache, cache: Arc::new(Mutex::new(None)),
        })
    }
    
    fn get_state(&self) -> Option<Tensor> { self.cache.lock().unwrap().clone() }
    fn set_state(&self, state: Tensor) { *self.cache.lock().unwrap() = Some(state); }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let bcx = self.in_proj.forward(x)?;
        let chunks = bcx.chunk(3, 2)?;
        let (b_param, c_param, x_param) = (&chunks[0], &chunks[1], &chunks[2]);
        let bx = (b_param * x_param)?; 
        let bx_t = bx.transpose(1, 2)?; 

        let conv_out = if seq_len > 1 {
            let padding = Tensor::zeros((b_sz, bx_t.dim(1)?, self.l_cache - 1), bx.dtype(), bx.device())?;
            let padded = Tensor::cat(&[&padding, &bx_t], 2)?;
            padded.conv1d(&self.conv_weight, 0, 1, 1, self.conv_weight.dim(0)?)?
        } else {
            let mut cache = self.cache.lock().unwrap();
            let state = match cache.as_ref() { Some(s) => s.clone(), None => Tensor::zeros((b_sz, bx_t.dim(1)?, self.l_cache), bx.dtype(), bx.device())? };
            let new_state = Tensor::cat(&[&state.i((.., .., 1..))?, &bx_t], 2)?;
            *cache = Some(new_state.clone());
            let w = self.conv_weight.reshape((1, (), self.l_cache))?; 
            new_state.broadcast_mul(&w)?.sum(2)?.unsqueeze(2)?
        };
        let mut out = conv_out;
        if let Some(bias) = &self.conv_bias { out = out.broadcast_add(&bias.reshape((1, (), 1))?)?; }
        let y = (c_param * out.transpose(1, 2)?)?;
        self.out_proj.forward(&y)
    }
}

struct Attention {
    q_proj: QLinear, k_proj: QLinear, v_proj: QLinear, o_proj: QLinear,
    q_norm: RmsNorm, k_norm: RmsNorm,
    n_heads: usize, n_kv_heads: usize, head_dim: usize, rope: RotaryEmbedding,
    kv_cache: Arc<Mutex<Option<(Tensor, Tensor)>>>,
}
impl Attention {
    fn new(cfg: &Config, weights: &mut Weights, prefix: &str, device: &Device, gguf: bool) -> Result<Self> {
        let rope = RotaryEmbedding::new(cfg.rope_theta, cfg.head_dim, 4096, device)?; 
        let (q, k, v, o, qn, kn) = if gguf {
            (format!("{}.attn_q.weight", prefix), format!("{}.attn_k.weight", prefix), format!("{}.attn_v.weight", prefix), format!("{}.attn_output.weight", prefix), format!("{}.attn_q_norm.weight", prefix), format!("{}.attn_k_norm.weight", prefix))
        } else {
            (format!("{}.self_attn.q_proj.weight", prefix), format!("{}.self_attn.k_proj.weight", prefix), format!("{}.self_attn.v_proj.weight", prefix), format!("{}.self_attn.out_proj.weight", prefix), format!("{}.self_attn.q_layernorm.weight", prefix), format!("{}.self_attn.k_layernorm.weight", prefix))
        };

        let q_proj = weights.pop_linear(&q)?;
        let k_proj = weights.pop_linear(&k)?;
        let v_proj = weights.pop_linear(&v)?;
        let o_proj = weights.pop_linear(&o)?;

        // AUTO-DETECT KV HEADS
        // If metadata is wrong (common in GGUF), we check the actual weight shape.
        // k_proj output dim = n_kv_heads * head_dim
        let mut n_kv_heads = cfg.num_key_value_heads;
        let inferred_kv = k_proj.out_dim / cfg.head_dim;
        if inferred_kv != n_kv_heads && k_proj.out_dim % cfg.head_dim == 0 {
            if prefix.contains(".0") || prefix.contains("blk.0") {
                eprintln!("Warning: Config n_kv_heads ({}) != Weights ({}). Using weights.", n_kv_heads, inferred_kv);
            }
            n_kv_heads = inferred_kv;
        }

        Ok(Self {
            q_proj, k_proj, v_proj, o_proj,
            q_norm: RmsNorm::new(weights.pop_tensor(&qn)?, cfg.norm_eps),
            k_norm: RmsNorm::new(weights.pop_tensor(&kn)?, cfg.norm_eps),
            n_heads: cfg.num_attention_heads, n_kv_heads, head_dim: cfg.head_dim, rope,
            kv_cache: Arc::new(Mutex::new(None)),
        })
    }
    fn get_state(&self) -> Option<(Tensor, Tensor)> { self.kv_cache.lock().unwrap().clone() }
    fn set_state(&self, k: Tensor, v: Tensor) { *self.kv_cache.lock().unwrap() = Some((k, v)); }
    fn forward(&self, x: &Tensor, pos: usize) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        
        // Ensure input is contiguous before projection (Helps AVX prefetching)
        let x = x.contiguous()?; 

        let q = self.q_norm.forward(&self.q_proj.forward(&x)?.reshape((b_sz, seq_len, self.n_heads, self.head_dim))?.transpose(1, 2)?)?;
        let k = self.k_norm.forward(&self.k_proj.forward(&x)?.reshape((b_sz, seq_len, self.n_kv_heads, self.head_dim))?.transpose(1, 2)?)?;
        let v = self.v_proj.forward(&x)?.reshape((b_sz, seq_len, self.n_kv_heads, self.head_dim))?.transpose(1, 2)?;

        let q = self.rope.apply(&q, pos)?.contiguous()?; // Force contiguous after rope
        let k = self.rope.apply(&k, pos)?.contiguous()?;

        let (k, v) = {
            let mut cache = self.kv_cache.lock().unwrap();
            match cache.as_ref() {
                Some((pk, pv)) => { 
                    let nk = Tensor::cat(&[pk, &k], 2)?.contiguous()?; // Force contiguous KV cache
                    let nv = Tensor::cat(&[pv, &v], 2)?.contiguous()?;
                    *cache = Some((nk.clone(), nv.clone()));
                    (nk, nv)
                }
                None => {
                    *cache = Some((k.clone(), v.clone()));
                    (k, v)
                }
            }
        };
        let n_rep = self.n_heads / self.n_kv_heads;
        let k = if n_rep > 1 { k.repeat((1, 1, n_rep, 1))?.reshape((b_sz, self.n_heads, k.dim(2)?, self.head_dim))? } else { k };
        let v = if n_rep > 1 { v.repeat((1, 1, n_rep, 1))?.reshape((b_sz, self.n_heads, v.dim(2)?, self.head_dim))? } else { v };
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = candle_nn::ops::softmax(&(q.matmul(&k.transpose(2, 3)?)? * scale)?, D::Minus1)?; 
        self.o_proj.forward(&attn_weights.matmul(&v)?.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?)
    }
}

enum Layer { Attn(Attention), Conv(ShortConv) }

pub struct Lfm2Model {
    embed: Embedding, layers: Vec<(Layer, RmsNorm, Mlp, RmsNorm)>, final_norm: RmsNorm, lm_head: QLinear,
}

impl Lfm2Model {
    pub fn load(path: &str, device: &Device) -> Result<Self> {
        let mut file = File::open(path)?;
        let content = gguf_file::Content::read(&mut file)?;
        let cfg = Config::from_gguf(&content)?;
        let mut weights = Weights::load(&mut file, &content, device)?;

        let embed_w = weights.pop_tensor("model.embed_tokens.weight")?;
        let embed = Embedding::new(embed_w.clone(), cfg.hidden_size);

        let use_gguf = weights.has("blk.0.attn_q.weight") || weights.has("blk.0.shortconv.in_proj.weight") || weights.has("blk.0.ffn_norm.weight");
        
        let mut layers = Vec::new();
        for i in 0..cfg.num_hidden_layers {
            let (p_base, p_norm1, p_norm2) = if use_gguf {
                (format!("blk.{}", i), format!("blk.{}.attn_norm.weight", i), format!("blk.{}.ffn_norm.weight", i))
            } else {
                (format!("model.layers.{}", i), format!("model.layers.{}.operator_norm.weight", i), format!("model.layers.{}.ffn_norm.weight", i))
            };

            let is_attn = if !cfg.layer_types.is_empty() {
                cfg.layer_types[i] == "full_attention"
            } else {
                if use_gguf { weights.has(&format!("{}.attn_q.weight", p_base)) } else { weights.has(&format!("{}.self_attn.q_proj.weight", p_base)) }
            };

            let core = if is_attn {
                Layer::Attn(Attention::new(&cfg, &mut weights, &p_base, device, use_gguf)?)
            } else {
                Layer::Conv(ShortConv::new(&cfg, &mut weights, &p_base, use_gguf)?)
            };

            let (w1, w2, w3) = if use_gguf {
                (format!("{}.ffn_gate.weight", p_base), format!("{}.ffn_down.weight", p_base), format!("{}.ffn_up.weight", p_base))
            } else {
                (format!("{}.feed_forward.w1.weight", p_base), format!("{}.feed_forward.w2.weight", p_base), format!("{}.feed_forward.w3.weight", p_base))
            };

            layers.push((
                core,
                RmsNorm::new(weights.pop_tensor(&p_norm1)?, cfg.norm_eps),
                Mlp { w1: weights.pop_linear(&w1)?, w2: weights.pop_linear(&w2)?, w3: weights.pop_linear(&w3)? },
                RmsNorm::new(weights.pop_tensor(&p_norm2)?, cfg.norm_eps),
            ));
        }

        let final_norm = RmsNorm::new(weights.pop_tensor("model.embedding_norm.weight")?, cfg.norm_eps);
        let lm_head = if weights.has("lm_head.weight") || weights.has("output.weight") {
            weights.pop_linear(if use_gguf { "output.weight" } else { "lm_head.weight" })?
        } else {
            // Re-use embed if head is tied (Standard LLaMA behavior)
            QLinear { inner: QLinearInner::Standard(Linear::new(embed_w, None)), out_dim: cfg.hidden_size }
        };

        Ok(Self { embed, layers, final_norm, lm_head })
    }
    // ... rest same ...
    pub fn reset_internal_state(&self) { for (l, _, _, _) in &self.layers { match l { Layer::Attn(a) => *a.kv_cache.lock().unwrap() = None, Layer::Conv(c) => *c.cache.lock().unwrap() = None } } }
    pub fn get_seq_len(&self) -> usize { for (l, _, _, _) in &self.layers { if let Layer::Attn(a) = l { if let Some((k, _)) = &*a.kv_cache.lock().unwrap() { return k.dim(2).unwrap_or(0); } } } 0 }
    pub fn save_state(&self, path: &str) -> Result<()> {
        let mut t = HashMap::new();
        for (i, (l, _, _, _)) in self.layers.iter().enumerate() { match l { Layer::Attn(a) => if let Some((k, v)) = a.get_state() { t.insert(format!("layer.{}.attn.k", i), k); t.insert(format!("layer.{}.attn.v", i), v); }, Layer::Conv(c) => if let Some(s) = c.get_state() { t.insert(format!("layer.{}.conv.state", i), s); } } }
        candle_core::safetensors::save(&t, path)
    }
    pub fn load_state(&self, path: &str, device: &Device) -> Result<()> {
        let t = candle_core::safetensors::load(path, device)?;
        self.reset_internal_state();
        for (i, (l, _, _, _)) in self.layers.iter().enumerate() { match l { Layer::Attn(a) => if let (Some(k), Some(v)) = (t.get(&format!("layer.{}.attn.k", i)), t.get(&format!("layer.{}.attn.v", i))) { a.set_state(k.clone(), v.clone()) }, Layer::Conv(c) => if let Some(s) = t.get(&format!("layer.{}.conv.state", i)) { c.set_state(s.clone()) } } }
        Ok(())
    }
    pub fn forward(&self, x: &Tensor, pos: usize) -> Result<Tensor> {
        let mut h = self.embed.forward(x)?;
        for (layer, op_norm, mlp, ffn_norm) in &self.layers {
            let resid = h.clone();
            let norm = op_norm.forward(&h)?;
            let out = match layer { Layer::Attn(a) => a.forward(&norm, pos)?, Layer::Conv(c) => c.forward(&norm)? };
            h = (resid + out)?;
            let resid = h.clone();
            let norm = ffn_norm.forward(&h)?;
            h = (resid + mlp.forward(&norm)?)?;
        }
        let h = self.final_norm.forward(&h)?;
        let (_b, seq, _) = h.dims3()?;
        self.lm_head.forward(&h.i((.., seq-1, ..))?)
    }
}