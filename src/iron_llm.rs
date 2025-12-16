//! # Iron-LLM: Language Model on BCC Lattice
//!
//! The key challenge: language is sequential, iron is configurational.
//! 
//! Our approach:
//! 1. Encode context (tokens) as spin patterns on input plane
//! 2. Let BCC lattice settle to stable configuration
//! 3. Read output plane as next-token prediction
//! 
//! The lattice layers act like transformer layers, but instead of
//! attention + FFN, we have BCC annealing + inter-layer coupling.
//!
//! ## Why This Might Work
//!
//! - Attention is about finding relevant context → Similar to associative memory
//! - FFN is about transforming representations → Similar to spin configuration changes
//! - The lattice naturally enforces consistency across the entire context
//!
//! ## The Iron Difference
//!
//! - Transformers: process tokens left-to-right, attend back
//! - Iron-LLM: settle entire context simultaneously, all positions interact

use crate::lattice::BCCLattice;
use crate::spin::Spin;
use crate::anneal::Annealer;
use crate::energy::Energy;

/// Embedding dimension (spins per token)
pub const EMBED_DIM: usize = 64;

/// Maximum context length (tokens)
pub const MAX_CONTEXT: usize = 128;

/// Token embedding: maps token IDs to spin patterns
#[derive(Clone)]
pub struct Embedding {
    /// Embedding matrix: vocab_size x embed_dim
    /// Each row is a spin pattern for a token
    pub weights: Vec<Vec<f32>>,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Embedding dimension
    pub embed_dim: usize,
}

impl Embedding {
    /// Create random embeddings for a vocabulary
    pub fn new(vocab_size: usize, embed_dim: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let weights: Vec<Vec<f32>> = (0..vocab_size)
            .map(|_| {
                (0..embed_dim)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect()
            })
            .collect();
        
        Self {
            weights,
            vocab_size,
            embed_dim,
        }
    }
    
    /// Convert embedding vector to spin pattern
    pub fn to_spins(&self, embedding: &[f32]) -> Vec<Spin> {
        embedding.iter()
            .map(|&x| if x >= 0.0 { Spin::Up } else { Spin::Down })
            .collect()
    }
    
    /// Convert spin pattern to embedding vector
    pub fn from_spins(&self, spins: &[Spin]) -> Vec<f32> {
        spins.iter()
            .map(|s| s.value())
            .collect()
    }
    
    /// Embed a token ID to a spin pattern
    pub fn embed(&self, token_id: usize) -> Vec<Spin> {
        if token_id < self.vocab_size {
            self.to_spins(&self.weights[token_id])
        } else {
            // Unknown token: random pattern
            vec![Spin::random(); self.embed_dim]
        }
    }
    
    /// Find nearest token to a spin pattern (for decoding)
    pub fn nearest_token(&self, spins: &[Spin]) -> usize {
        let query: Vec<f32> = spins.iter().map(|s| s.value()).collect();
        
        let mut best_idx = 0;
        let mut best_sim = f32::NEG_INFINITY;
        
        for (idx, embedding) in self.weights.iter().enumerate() {
            let sim: f32 = query.iter()
                .zip(embedding.iter())
                .map(|(q, e)| q * e)
                .sum();
            
            if sim > best_sim {
                best_sim = sim;
                best_idx = idx;
            }
        }
        
        best_idx
    }
}

/// A single layer of the Iron-LLM
/// 
/// Each layer is a BCC lattice that:
/// 1. Receives input spins on one face
/// 2. Settles to stable configuration
/// 3. Outputs spins on opposite face
pub struct IronLayer {
    /// The BCC lattice for this layer
    pub lattice: BCCLattice,
    /// Width (tokens in context)
    pub width: usize,
    /// Height (embedding dimension)
    pub height: usize,
    /// Depth of the lattice (processing depth)
    pub depth: usize,
    /// Temperature for annealing
    pub temperature: f32,
    /// Number of annealing steps per forward pass
    pub anneal_steps: usize,
}

impl IronLayer {
    /// Create a new iron layer
    pub fn new(context_len: usize, embed_dim: usize, depth: usize) -> Self {
        let mut lattice = BCCLattice::new(context_len, embed_dim, depth);
        
        // Initialize couplings based on position (like positional encoding)
        // Closer positions have stronger coupling
        for i in 0..lattice.cells.len() {
            let (x1, y1, z1) = lattice.coords(i);
            let neighbors = lattice.cells[i].neighbors;
            
            for (k, &j) in neighbors.iter().enumerate() {
                let (x2, y2, z2) = lattice.coords(j);
                
                // Distance-based coupling (positions closer together couple stronger)
                let dx = (x1 as f32 - x2 as f32).abs();
                let dist = 1.0 + dx * 0.1;  // Decay with token distance
                lattice.cells[i].coupling[k] = 1.0 / dist;
            }
        }
        
        Self {
            lattice,
            width: context_len,
            height: embed_dim,
            depth,
            temperature: 1.0,
            anneal_steps: 1000,
        }
    }
    
    /// Set input spins (z=0 plane)
    pub fn set_input(&mut self, tokens: &[Vec<Spin>]) {
        for (x, token_spins) in tokens.iter().enumerate() {
            for (y, &spin) in token_spins.iter().enumerate() {
                if x < self.width && y < self.height {
                    let idx = self.lattice.index(x, y, 0);
                    self.lattice.cells[idx].spin = spin;
                    // Pin input spins with strong bias
                    self.lattice.cells[idx].bias = 10.0 * spin.value();
                }
            }
        }
    }
    
    /// Get output spins (z=depth-1 plane)
    pub fn get_output(&self) -> Vec<Vec<Spin>> {
        let z = self.depth - 1;
        let mut output = Vec::with_capacity(self.width);
        
        for x in 0..self.width {
            let mut token_out = Vec::with_capacity(self.height);
            for y in 0..self.height {
                let idx = self.lattice.index(x, y, z);
                token_out.push(self.lattice.cells[idx].spin);
            }
            output.push(token_out);
        }
        
        output
    }
    
    /// Forward pass: set input, anneal, read output
    pub fn forward(&mut self, input: &[Vec<Spin>]) -> Vec<Vec<Spin>> {
        // Set input
        self.set_input(input);
        
        // Anneal to stable configuration
        let mut annealer = Annealer::new(self.temperature, self.anneal_steps);
        annealer.anneal(&mut self.lattice);
        
        // Read output
        self.get_output()
    }
    
    /// Clear biases (for next forward pass)
    pub fn clear_input(&mut self) {
        for cell in &mut self.lattice.cells {
            cell.bias = 0.0;
        }
    }
}

/// The full Iron-LLM model
pub struct IronLLM {
    /// Token embeddings
    pub embedding: Embedding,
    /// Stack of iron layers
    pub layers: Vec<IronLayer>,
    /// Output projection (for next-token prediction)
    pub output_proj: Vec<Vec<f32>>,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Context length
    pub context_len: usize,
    /// Embedding dimension
    pub embed_dim: usize,
}

impl IronLLM {
    /// Create a new Iron-LLM
    pub fn new(
        vocab_size: usize,
        context_len: usize,
        embed_dim: usize,
        num_layers: usize,
        layer_depth: usize,
    ) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let embedding = Embedding::new(vocab_size, embed_dim);
        
        let layers: Vec<IronLayer> = (0..num_layers)
            .map(|_| IronLayer::new(context_len, embed_dim, layer_depth))
            .collect();
        
        // Random output projection
        let output_proj: Vec<Vec<f32>> = (0..vocab_size)
            .map(|_| {
                (0..embed_dim)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect()
            })
            .collect();
        
        Self {
            embedding,
            layers,
            output_proj,
            vocab_size,
            context_len,
            embed_dim,
        }
    }
    
    /// Embed tokens to spin patterns
    pub fn embed_tokens(&self, tokens: &[usize]) -> Vec<Vec<Spin>> {
        tokens.iter()
            .map(|&t| self.embedding.embed(t))
            .collect()
    }
    
    /// Forward pass: tokens → next token logits
    pub fn forward(&mut self, tokens: &[usize]) -> Vec<f32> {
        // Embed tokens
        let mut hidden = self.embed_tokens(tokens);
        
        // Pass through each layer
        for layer in &mut self.layers {
            layer.clear_input();
            hidden = layer.forward(&hidden);
        }
        
        // Get last position's output
        let last_hidden = if let Some(last) = hidden.last() {
            self.embedding.from_spins(last)
        } else {
            vec![0.0; self.embed_dim]
        };
        
        // Project to vocabulary (logits)
        let mut logits = vec![0.0; self.vocab_size];
        for (i, proj) in self.output_proj.iter().enumerate() {
            logits[i] = last_hidden.iter()
                .zip(proj.iter())
                .map(|(h, p)| h * p)
                .sum();
        }
        
        logits
    }
    
    /// Sample next token from logits
    pub fn sample(&self, logits: &[f32], temperature: f32) -> usize {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // Apply temperature
        let scaled: Vec<f32> = logits.iter()
            .map(|&l| l / temperature)
            .collect();
        
        // Softmax
        let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scaled.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
        
        // Sample from distribution
        let r: f32 = rng.gen();
        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return i;
            }
        }
        
        probs.len() - 1
    }
    
    /// Generate tokens autoregressively
    pub fn generate(&mut self, prompt: &[usize], max_tokens: usize, temperature: f32) -> Vec<usize> {
        let mut tokens = prompt.to_vec();
        
        for _ in 0..max_tokens {
            // Truncate to context length
            let context: Vec<usize> = if tokens.len() > self.context_len {
                tokens[tokens.len() - self.context_len..].to_vec()
            } else {
                tokens.clone()
            };
            
            // Forward pass
            let logits = self.forward(&context);
            
            // Sample next token
            let next = self.sample(&logits, temperature);
            tokens.push(next);
        }
        
        tokens
    }
}

/// Simple character-level tokenizer for testing
pub struct CharTokenizer {
    pub char_to_id: std::collections::HashMap<char, usize>,
    pub id_to_char: Vec<char>,
}

impl CharTokenizer {
    /// Create tokenizer from a string (uses all unique characters)
    pub fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();
        
        let char_to_id: std::collections::HashMap<char, usize> = chars
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();
        
        Self {
            char_to_id,
            id_to_char: chars,
        }
    }
    
    /// Tokenize a string
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .filter_map(|c| self.char_to_id.get(&c).copied())
            .collect()
    }
    
    /// Detokenize a sequence
    pub fn decode(&self, tokens: &[usize]) -> String {
        tokens.iter()
            .filter_map(|&t| self.id_to_char.get(t))
            .collect()
    }
    
    /// Vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.id_to_char.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_embedding() {
        let emb = Embedding::new(100, 64);
        let spins = emb.embed(42);
        assert_eq!(spins.len(), 64);
        
        let nearest = emb.nearest_token(&spins);
        assert_eq!(nearest, 42);  // Should find itself
    }
    
    #[test]
    fn test_iron_layer() {
        let mut layer = IronLayer::new(8, 16, 4);
        
        // Random input
        let input: Vec<Vec<Spin>> = (0..8)
            .map(|_| (0..16).map(|_| Spin::random()).collect())
            .collect();
        
        let output = layer.forward(&input);
        assert_eq!(output.len(), 8);
        assert_eq!(output[0].len(), 16);
    }
    
    #[test]
    fn test_tokenizer() {
        let tok = CharTokenizer::from_text("hello world");
        let encoded = tok.encode("hello");
        let decoded = tok.decode(&encoded);
        assert_eq!(decoded, "hello");
    }
}
