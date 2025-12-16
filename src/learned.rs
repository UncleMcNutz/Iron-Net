//! # Learned Embeddings: Data-Driven Word→Form Mappings
//!
//! Instead of hand-crafted rules for word→form,
//! learn the mappings from data.
//!
//! ## The Approach
//!
//! 1. Start with random forms for each word
//! 2. Words that appear in similar contexts → similar forms
//! 3. Co-occurring words → correlated forms
//! 4. Forms settle into meaningful geometry through training
//!
//! ## Distributional Semantics in 3D
//!
//! "You shall know a word by the company it keeps" - Firth
//! Words in similar contexts get similar 3D forms.

use crate::concept::{Concept, ConceptMind};
use crate::spin::Spin;
use crate::algebra::ConceptAlgebra;
use std::collections::HashMap;
use rand::Rng;

/// Learned concept embedding: maps tokens to learned 3D forms
pub struct LearnedEmbedding {
    /// Embedding dimension (lattice size)
    pub size: usize,
    /// Token to form mapping
    pub embeddings: HashMap<String, Concept>,
    /// Token frequencies (for weighting)
    pub frequencies: HashMap<String, usize>,
    /// Co-occurrence counts (for learning)
    pub cooccurrences: HashMap<(String, String), f32>,
    /// Learning rate
    pub learning_rate: f32,
}

impl LearnedEmbedding {
    pub fn new(size: usize) -> Self {
        Self {
            size,
            embeddings: HashMap::new(),
            frequencies: HashMap::new(),
            cooccurrences: HashMap::new(),
            learning_rate: 0.1,
        }
    }
    
    /// Get or create embedding for token
    pub fn get_or_create(&mut self, token: &str) -> Concept {
        if let Some(concept) = self.embeddings.get(token) {
            return concept.clone();
        }
        
        // Initialize with random form
        let concept = self.random_concept(token);
        self.embeddings.insert(token.to_string(), concept.clone());
        self.frequencies.insert(token.to_string(), 0);
        
        concept
    }
    
    /// Get embedding (None if not learned)
    pub fn get(&self, token: &str) -> Option<&Concept> {
        self.embeddings.get(token)
    }
    
    /// Create random concept seeded by token
    fn random_concept(&self, token: &str) -> Concept {
        let mut rng = rand::thread_rng();
        let hash = token.bytes().fold(0u64, |a, b| a.wrapping_mul(31).wrapping_add(b as u64));
        
        // Use hash to seed random state somewhat deterministically
        let density = 0.2 + (hash % 100) as f32 / 200.0;  // 0.2 - 0.7
        
        let mut form = vec![Spin::Down; self.size * self.size * self.size];
        for i in 0..form.len() {
            if rng.gen::<f32>() < density {
                form[i] = Spin::Up;
            }
        }
        
        Concept {
            form,
            dims: (self.size, self.size, self.size),
            label: Some(token.to_string()),
        }
    }
    
    /// Build vocabulary from text
    pub fn build_vocabulary(&mut self, text: &str) {
        let tokens: Vec<&str> = text.split_whitespace().collect();
        
        for token in &tokens {
            let token_str = token.to_lowercase();
            self.get_or_create(&token_str);
            *self.frequencies.entry(token_str).or_insert(0) += 1;
        }
    }
    
    /// Count co-occurrences within window
    pub fn count_cooccurrences(&mut self, text: &str, window: usize) {
        let tokens: Vec<String> = text.split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();
        
        for i in 0..tokens.len() {
            for j in 1..=window {
                if i + j < tokens.len() {
                    let pair = (tokens[i].clone(), tokens[i + j].clone());
                    let weight = 1.0 / j as f32;  // Closer = more weight
                    *self.cooccurrences.entry(pair).or_insert(0.0) += weight;
                }
            }
        }
    }
    
    /// Train embeddings based on co-occurrence
    /// Words that appear together should have similar forms
    pub fn train_epoch(&mut self) {
        let algebra = ConceptAlgebra::new(self.size);
        
        // For each co-occurring pair, make forms more similar
        let pairs: Vec<_> = self.cooccurrences.iter()
            .map(|((a, b), c)| (a.clone(), b.clone(), *c))
            .collect();
        
        for (token_a, token_b, cooc) in pairs {
            if cooc < 1.0 { continue; }
            
            let concept_a = match self.embeddings.get(&token_a) {
                Some(c) => c.clone(),
                None => continue,
            };
            let concept_b = match self.embeddings.get(&token_b) {
                Some(c) => c.clone(),
                None => continue,
            };
            
            // Move forms toward each other proportional to co-occurrence
            let strength = (cooc.ln() * self.learning_rate).min(0.3);
            
            // Blend A toward B
            let new_a = concept_a.blend(&concept_b, strength);
            // Blend B toward A
            let new_b = concept_b.blend(&concept_a, strength);
            
            self.embeddings.insert(token_a, new_a);
            self.embeddings.insert(token_b, new_b);
        }
    }
    
    /// Train for multiple epochs
    pub fn train(&mut self, text: &str, window: usize, epochs: usize) {
        self.build_vocabulary(text);
        self.count_cooccurrences(text, window);
        
        for _ in 0..epochs {
            self.train_epoch();
        }
    }
    
    /// Find most similar tokens to a given form
    pub fn most_similar(&self, concept: &Concept, k: usize) -> Vec<(String, f32)> {
        let mut similarities: Vec<_> = self.embeddings.iter()
            .map(|(token, emb)| (token.clone(), concept.similarity(emb)))
            .collect();
        
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(k);
        similarities
    }
    
    /// Find most similar tokens to a given token
    pub fn similar_to(&self, token: &str, k: usize) -> Vec<(String, f32)> {
        if let Some(concept) = self.embeddings.get(token) {
            self.most_similar(concept, k + 1)
                .into_iter()
                .filter(|(t, _)| t != token)
                .take(k)
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Analogy: A is to B as C is to ?
    pub fn analogy(&self, a: &str, b: &str, c: &str, k: usize) -> Vec<(String, f32)> {
        let algebra = ConceptAlgebra::new(self.size);
        
        let ca = match self.embeddings.get(a) { Some(c) => c, None => return Vec::new() };
        let cb = match self.embeddings.get(b) { Some(c) => c, None => return Vec::new() };
        let cc = match self.embeddings.get(c) { Some(c) => c, None => return Vec::new() };
        
        let result = algebra.analogy(ca, cb, cc);
        
        self.most_similar(&result, k + 3)
            .into_iter()
            .filter(|(t, _)| t != a && t != b && t != c)
            .take(k)
            .collect()
    }
    
    /// Vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.embeddings.len()
    }
}

/// Demonstrate learned embeddings
pub fn demo_learned_embeddings() {
    use crate::concept::visualize_concept;
    
    println!("================================================================");
    println!("      LEARNED EMBEDDINGS: Data-Driven Word→Form Mappings");
    println!("================================================================");
    println!();
    
    let size = 10;
    let mut embeddings = LearnedEmbedding::new(size);
    
    // Training corpus
    let corpus = "
        the iron lattice thinks in configurations
        iron is a metal that forms crystals
        the metal lattice has magnetic properties
        thoughts are configurations in the mind
        the mind thinks in abstract patterns
        patterns form in the crystal lattice
        iron and metal share magnetic properties
        the abstract mind forms concrete thoughts
        configurations settle into stable patterns
        stable patterns are the ground state
        iron speaks in forms not in words
        forms are stable configurations
        the lattice is made of iron atoms
        atoms form patterns in the crystal
        crystal structures are stable configurations
    ";
    
    println!("Training on corpus ({} chars)...\n", corpus.len());
    
    // Train
    embeddings.train(corpus, 5, 10);
    
    println!("Vocabulary size: {} tokens\n", embeddings.vocab_size());
    
    // Show some learned embeddings
    println!("=== LEARNED FORMS ===\n");
    
    for word in &["iron", "lattice", "mind", "patterns"] {
        if let Some(concept) = embeddings.get(word) {
            println!("'{}':", word);
            visualize_concept(concept);
            println!();
        }
    }
    
    // Similarity search
    println!("=== SIMILARITY SEARCH ===\n");
    
    for word in &["iron", "mind", "patterns"] {
        let similar = embeddings.similar_to(word, 5);
        println!("Similar to '{}':", word);
        for (w, sim) in similar {
            println!("  {}: {:.1}%", w, sim * 100.0);
        }
        println!();
    }
    
    // Analogies
    println!("=== ANALOGIES ===\n");
    
    let analogies = [
        ("iron", "metal", "mind"),  // iron:metal :: mind:?
        ("lattice", "crystal", "thoughts"),  // lattice:crystal :: thoughts:?
        ("forms", "stable", "patterns"),  // forms:stable :: patterns:?
    ];
    
    for (a, b, c) in &analogies {
        let results = embeddings.analogy(a, b, c, 3);
        println!("{}:{} :: {}:?", a, b, c);
        for (w, sim) in results {
            println!("  → {} ({:.1}%)", w, sim * 100.0);
        }
        println!();
    }
    
    println!("=== THE INSIGHT ===\n");
    println!("Words that appear in similar contexts → similar 3D forms.");
    println!("'iron' and 'metal' co-occur → their forms converge.");
    println!("'mind' and 'thinks' co-occur → related geometry.");
    println!();
    println!("Meaning emerges from usage patterns, rendered in 3D.");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cooccurrence_increases_similarity() {
        let mut emb = LearnedEmbedding::new(8);
        
        // Create two random embeddings
        emb.get_or_create("word1");
        emb.get_or_create("word2");
        
        let sim_before = emb.get("word1").unwrap().similarity(emb.get("word2").unwrap());
        
        // Add co-occurrences
        emb.cooccurrences.insert(("word1".to_string(), "word2".to_string()), 10.0);
        
        // Train
        emb.train_epoch();
        
        let sim_after = emb.get("word1").unwrap().similarity(emb.get("word2").unwrap());
        
        // Similarity should increase
        assert!(sim_after >= sim_before);
    }
}
