//! # Iron Training: Learning Through Energy Landscape Sculpting
//!
//! Traditional neural nets learn through gradient descent:
//! - Compute loss
//! - Backpropagate gradients
//! - Update weights
//!
//! Iron-LLM learns differently:
//! - Compute loss (what patterns should be stable?)
//! - Adjust couplings so correct outputs become energy minima
//! - The lattice naturally settles to trained patterns
//!
//! This is contrastive Hebbian learning + Boltzmann machine training.
//! The key insight: make the correct output the STABLE configuration.

use crate::lattice::BCCLattice;
use crate::spin::Spin;
use crate::iron_llm::{IronLLM, IronLayer, CharTokenizer};
use crate::anneal::Annealer;

/// Training configuration
#[derive(Clone)]
pub struct TrainConfig {
    /// Learning rate for coupling updates
    pub learning_rate: f32,
    /// Temperature for the "free" phase (without clamped output)
    pub free_temp: f32,
    /// Temperature for the "clamped" phase (with clamped output)
    pub clamped_temp: f32,
    /// Annealing steps per phase
    pub anneal_steps: usize,
    /// Weight decay (regularization)
    pub weight_decay: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            free_temp: 1.0,
            clamped_temp: 0.5,
            anneal_steps: 500,
            weight_decay: 0.001,
        }
    }
}

/// Training statistics
#[derive(Clone, Default)]
pub struct TrainStats {
    pub total_loss: f32,
    pub accuracy: f32,
    pub samples: usize,
    pub correct: usize,
}

impl TrainStats {
    pub fn reset(&mut self) {
        self.total_loss = 0.0;
        self.accuracy = 0.0;
        self.samples = 0;
        self.correct = 0;
    }
    
    pub fn update(&mut self, loss: f32, correct: bool) {
        self.total_loss += loss;
        self.samples += 1;
        if correct { self.correct += 1; }
        self.accuracy = self.correct as f32 / self.samples as f32;
    }
}

/// Iron trainer using Contrastive Hebbian Learning
/// 
/// The algorithm:
/// 1. CLAMPED PHASE: Set input AND correct output, let middle settle
/// 2. FREE PHASE: Set only input, let the whole thing settle
/// 3. UPDATE: Δw_ij = η * (<s_i * s_j>_clamped - <s_i * s_j>_free)
/// 
/// This makes the clamped (correct) configuration more stable than whatever
/// the network would naturally settle to.
pub struct IronTrainer {
    /// Training configuration
    pub config: TrainConfig,
    /// Training statistics
    pub stats: TrainStats,
}

impl IronTrainer {
    pub fn new(config: TrainConfig) -> Self {
        Self {
            config,
            stats: TrainStats::default(),
        }
    }
    
    /// Train on a single example (context -> next_token)
    /// 
    /// Returns (loss, predicted_token)
    pub fn train_step(
        &mut self,
        model: &mut IronLLM,
        context: &[usize],  // Input tokens
        target: usize,      // Correct next token
    ) -> (f32, usize) {
        // Embed input
        let input_spins = model.embed_tokens(context);
        let target_spins = model.embedding.embed(target);
        
        // For each layer, do contrastive learning
        for layer in &mut model.layers {
            self.train_layer(layer, &input_spins, &target_spins);
        }
        
        // Forward pass to get prediction
        let logits = model.forward(context);
        let predicted = argmax(&logits);
        
        // Cross-entropy loss (simplified)
        let loss = -logits[target] + log_sum_exp(&logits);
        
        let correct = predicted == target;
        self.stats.update(loss, correct);
        
        (loss, predicted)
    }
    
    /// Train a single layer using contrastive Hebbian learning
    fn train_layer(
        &self,
        layer: &mut IronLayer,
        input: &[Vec<Spin>],
        target: &[Spin],
    ) {
        let config = &self.config;
        let z_out = layer.depth - 1;
        
        // ===== CLAMPED PHASE =====
        // Set input (z=0) and output (z=depth-1), let middle settle
        layer.clear_input();
        layer.set_input(input);
        
        // Clamp output layer to target
        for (y, &spin) in target.iter().enumerate() {
            if y < layer.height {
                // Clamp all x positions to same target (last token's target)
                let x = layer.width - 1;  // Last position
                let idx = layer.lattice.index(x, y, z_out);
                layer.lattice.cells[idx].spin = spin;
                layer.lattice.cells[idx].bias = 20.0 * spin.value();
            }
        }
        
        // Anneal with clamped boundaries
        let mut annealer = Annealer::new(config.clamped_temp, config.anneal_steps);
        annealer.anneal(&mut layer.lattice);
        
        // Record clamped correlations
        let clamped_corr = self.compute_correlations(&layer.lattice);
        
        // ===== FREE PHASE =====
        // Set only input, let output be free
        layer.clear_input();
        layer.set_input(input);
        // Don't clamp output this time
        
        let mut annealer = Annealer::new(config.free_temp, config.anneal_steps);
        annealer.anneal(&mut layer.lattice);
        
        // Record free correlations
        let free_corr = self.compute_correlations(&layer.lattice);
        
        // ===== UPDATE COUPLINGS =====
        // Δw_ij = η * (clamped_corr - free_corr)
        for i in 0..layer.lattice.cells.len() {
            let neighbors = layer.lattice.cells[i].neighbors;
            for k in 0..8 {
                let j = neighbors[k];
                let delta = config.learning_rate * (
                    clamped_corr[i][k] - free_corr[i][k]
                );
                
                // Apply update with weight decay
                let w = &mut layer.lattice.cells[i].coupling[k];
                *w = *w * (1.0 - config.weight_decay) + delta;
                
                // Keep couplings bounded
                *w = w.clamp(-10.0, 10.0);
            }
        }
    }
    
    /// Compute correlation <s_i * s_j> for all neighbor pairs
    fn compute_correlations(&self, lattice: &BCCLattice) -> Vec<[f32; 8]> {
        let mut corr = vec![[0.0f32; 8]; lattice.cells.len()];
        
        for i in 0..lattice.cells.len() {
            let s_i = lattice.cells[i].spin.value();
            let neighbors = lattice.cells[i].neighbors;
            
            for (k, &j) in neighbors.iter().enumerate() {
                let s_j = lattice.cells[j].spin.value();
                corr[i][k] = s_i * s_j;
            }
        }
        
        corr
    }
    
    /// Train on a corpus (string of text)
    pub fn train_epoch(
        &mut self,
        model: &mut IronLLM,
        tokenizer: &CharTokenizer,
        corpus: &str,
        context_len: usize,
    ) {
        let tokens = tokenizer.encode(corpus);
        
        if tokens.len() < context_len + 1 {
            println!("Corpus too short for training");
            return;
        }
        
        // Slide window through corpus
        for i in 0..(tokens.len() - context_len) {
            let context = &tokens[i..i + context_len];
            let target = tokens[i + context_len];
            
            let (loss, pred) = self.train_step(model, context, target);
            
            // Print progress every 50 steps
            if i % 50 == 0 && i > 0 {
                println!(
                    "  Step {}: loss={:.3}, acc={:.1}%",
                    i, self.stats.total_loss / self.stats.samples as f32,
                    self.stats.accuracy * 100.0
                );
            }
        }
    }
}

/// Argmax helper
fn argmax(slice: &[f32]) -> usize {
    slice.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Log-sum-exp for numerical stability
fn log_sum_exp(slice: &[f32]) -> f32 {
    let max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = slice.iter().map(|x| (x - max).exp()).sum();
    max + sum.ln()
}

/// Demonstration of training
pub fn demo_training(
    model: &mut IronLLM,
    tokenizer: &CharTokenizer,
    corpus: &str,
    epochs: usize,
) {
    println!("Training Iron-LLM with Contrastive Hebbian Learning");
    println!("====================================================");
    println!();
    
    let config = TrainConfig {
        learning_rate: 0.05,
        free_temp: 2.0,
        clamped_temp: 0.3,
        anneal_steps: 200,
        weight_decay: 0.0001,
    };
    
    let mut trainer = IronTrainer::new(config);
    
    for epoch in 0..epochs {
        println!("Epoch {}/{}", epoch + 1, epochs);
        trainer.stats.reset();
        
        trainer.train_epoch(model, tokenizer, corpus, model.context_len);
        
        println!(
            "  Final: loss={:.3}, accuracy={:.1}%\n",
            trainer.stats.total_loss / trainer.stats.samples as f32,
            trainer.stats.accuracy * 100.0
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0, 1.0, 2.0]), 0);
    }
    
    #[test]
    fn test_log_sum_exp() {
        let lse = log_sum_exp(&[1.0, 2.0, 3.0]);
        // Should be close to log(e^1 + e^2 + e^3) ≈ 3.41
        assert!((lse - 3.41).abs() < 0.1);
    }
}
