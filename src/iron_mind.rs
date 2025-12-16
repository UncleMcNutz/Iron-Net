//! # Iron Mind: The Complete Concept-Based Intelligence
//!
//! This unifies all components into a working system:
//! - Learned embeddings (word → form from data)
//! - Concept algebra (form operations: +, -, analogy)
//! - Hierarchical structure (forms within forms)
//! - Multi-modal input (text, images, audio → same forms)
//! - Iron thinking (3D lattice processing)
//!
//! ## The Full Architecture
//!
//! ```text
//!                    ┌─────────────┐
//!     Text ─────────→│             │
//!                    │   Encoder   │
//!     Image ────────→│  (learned)  │──→ 3D Concept
//!                    │             │         │
//!     Audio ────────→│             │         │
//!                    └─────────────┘         ▼
//!                                      ┌───────────┐
//!                                      │           │
//!                                      │   Iron    │
//!                                      │   Mind    │
//!                                      │  (BCC)    │
//!                                      │           │
//!                                      └─────┬─────┘
//!                                            │
//!                    ┌─────────────┐         │
//!     Text ←─────────│             │←────────┘
//!                    │   Decoder   │
//!     Image ←────────│             │
//!                    └─────────────┘
//! ```

use crate::concept::{Concept, ConceptMind, ConceptSculptor, visualize_concept};
use crate::algebra::ConceptAlgebra;
use crate::hierarchy::{HierarchicalConcept, HierarchicalParser, ConceptRole, print_hierarchy};
use crate::learned::LearnedEmbedding;
use crate::multimodal::MultiModalEncoder;
use crate::spin::Spin;
use std::collections::HashMap;

/// The Iron Mind: unified concept-based intelligence
pub struct IronMind {
    /// Size of concept lattice
    pub size: usize,
    /// The thinking substrate
    pub mind: ConceptMind,
    /// Learned word embeddings
    pub embeddings: LearnedEmbedding,
    /// Concept algebra operations
    pub algebra: ConceptAlgebra,
    /// Hierarchical parser
    pub parser: HierarchicalParser,
    /// Multi-modal encoder
    pub encoder: MultiModalEncoder,
    /// Context: recent concepts
    context: Vec<Concept>,
    /// Maximum context length
    max_context: usize,
}

impl IronMind {
    /// Create a new Iron Mind
    pub fn new(size: usize, max_context: usize) -> Self {
        Self {
            size,
            mind: ConceptMind::new(size),
            embeddings: LearnedEmbedding::new(size),
            algebra: ConceptAlgebra::new(size),
            parser: HierarchicalParser::new(size),
            encoder: MultiModalEncoder::new(size),
            context: Vec::new(),
            max_context,
        }
    }
    
    /// Train on text corpus
    pub fn train(&mut self, corpus: &str, epochs: usize) {
        // Learn word embeddings from co-occurrence
        self.embeddings.train(corpus, 5, epochs);
        
        // Also teach concepts as attractors
        let words: Vec<&str> = corpus.split_whitespace().collect();
        for word in &words {
            if let Some(concept) = self.embeddings.get(&word.to_lowercase()) {
                self.mind.learn(concept);
            }
        }
        
        // Learn word sequences as hierarchical patterns
        for window in words.windows(3) {
            let hierarchy = self.parser.parse(&window.join(" "));
            let flat = hierarchy.flatten();
            self.mind.learn(&flat);
        }
    }
    
    /// Process text input, return thought
    pub fn process_text(&mut self, input: &str) -> Concept {
        // Parse as hierarchy
        let hierarchy = self.parser.parse(input);
        let concept = hierarchy.flatten();
        
        // Add context influence
        let contextualized = self.contextualize(&concept);
        
        // Think
        let thought = self.mind.think(&contextualized, 2000);
        
        // Update context
        self.update_context(&thought);
        
        thought
    }
    
    /// Process image input
    pub fn process_image(&mut self, pixels: &[f32], width: usize, height: usize) -> Concept {
        let concept = self.encoder.encode_image(pixels, width, height);
        let contextualized = self.contextualize(&concept);
        let thought = self.mind.think(&contextualized, 2000);
        self.update_context(&thought);
        thought
    }
    
    /// Process audio input
    pub fn process_audio(&mut self, samples: &[f32], sample_rate: usize) -> Concept {
        let concept = self.encoder.encode_audio(samples, sample_rate);
        let contextualized = self.contextualize(&concept);
        let thought = self.mind.think(&contextualized, 2000);
        self.update_context(&thought);
        thought
    }
    
    /// Process numeric vector
    pub fn process_vector(&mut self, values: &[f32]) -> Concept {
        let concept = self.encoder.encode_vector(values);
        let contextualized = self.contextualize(&concept);
        let thought = self.mind.think(&contextualized, 2000);
        self.update_context(&thought);
        thought
    }
    
    /// Add context to a concept
    fn contextualize(&self, concept: &Concept) -> Concept {
        if self.context.is_empty() {
            return concept.clone();
        }
        
        // Blend with context (more recent = more weight)
        let mut result = concept.clone();
        for (i, ctx) in self.context.iter().enumerate() {
            let weight = (i + 1) as f32 / self.context.len() as f32 * 0.2;
            result = result.blend(ctx, weight);
        }
        
        result
    }
    
    /// Update context with new concept
    fn update_context(&mut self, concept: &Concept) {
        self.context.push(concept.clone());
        while self.context.len() > self.max_context {
            self.context.remove(0);
        }
    }
    
    /// Clear context
    pub fn reset(&mut self) {
        self.context.clear();
    }
    
    /// Decode concept to closest words
    pub fn decode_to_words(&self, concept: &Concept, k: usize) -> Vec<(String, f32)> {
        self.embeddings.most_similar(concept, k)
    }
    
    /// Decode concept to description
    pub fn describe(&self, concept: &Concept) -> String {
        let words = self.decode_to_words(concept, 5);
        let density = concept.density();
        let (sx, sy, sz) = concept.symmetry();
        
        let word_str = words.iter()
            .take(3)
            .map(|(w, s)| format!("{} ({:.0}%)", w, s * 100.0))
            .collect::<Vec<_>>()
            .join(", ");
        
        format!(
            "Form: {} | Density: {:.0}% | Symmetry: ({:.0}%, {:.0}%, {:.0}%)",
            word_str, density * 100.0, sx * 100.0, sy * 100.0, sz * 100.0
        )
    }
    
    // ========================
    // CONCEPT OPERATIONS
    // ========================
    
    /// Combine concepts
    pub fn combine(&self, a: &Concept, b: &Concept) -> Concept {
        self.algebra.add(a, b)
    }
    
    /// Difference between concepts
    pub fn difference(&self, a: &Concept, b: &Concept) -> Concept {
        self.algebra.subtract(a, b)
    }
    
    /// Analogy: A is to B as C is to ?
    pub fn analogy(&mut self, a: &Concept, b: &Concept, c: &Concept) -> Concept {
        let raw = self.algebra.analogy(a, b, c);
        self.mind.think(&raw, 3000)  // Stabilize
    }
    
    /// Word analogy
    pub fn word_analogy(&mut self, a: &str, b: &str, c: &str) -> Vec<(String, f32)> {
        self.embeddings.analogy(a, b, c, 5)
    }
    
    // ========================
    // GENERATION
    // ========================
    
    /// Generate continuation from concept
    pub fn generate(&mut self, seed: &Concept, steps: usize) -> Vec<Concept> {
        let mut sequence = vec![seed.clone()];
        let mut current = seed.clone();
        
        for _ in 0..steps {
            // Add noise for variation
            let sculptor = ConceptSculptor::new(self.size, self.size, self.size);
            let noise = sculptor.noise(0.1);
            let noisy = current.blend(&noise, 0.1);
            
            // Think to get next concept
            let next = self.mind.think(&noisy, 1500);
            sequence.push(next.clone());
            current = next;
        }
        
        sequence
    }
    
    /// Generate text from prompt
    pub fn generate_text(&mut self, prompt: &str, num_words: usize) -> String {
        let mut result = prompt.to_string();
        
        // Process prompt
        let seed = self.process_text(prompt);
        
        // Generate concepts
        let concepts = self.generate(&seed, num_words);
        
        // Decode each to word
        for concept in concepts.iter().skip(1) {
            let words = self.decode_to_words(concept, 3);
            if let Some((word, _)) = words.first() {
                result.push(' ');
                result.push_str(word);
            }
        }
        
        result
    }
    
    /// Dream: generate from random seed
    pub fn dream(&mut self, steps: usize) -> Vec<Concept> {
        let sculptor = ConceptSculptor::new(self.size, self.size, self.size);
        let seed = sculptor.noise(0.5);
        self.generate(&seed, steps)
    }
}

/// Full demonstration of Iron Mind
pub fn demo_iron_mind() {
    println!("================================================================");
    println!("            IRON MIND: Unified Concept Intelligence");
    println!("================================================================");
    println!();
    
    let size = 10;
    let max_context = 5;
    
    println!("Creating Iron Mind ({}x{}x{} lattice)...\n", size, size, size);
    let mut mind = IronMind::new(size, max_context);
    
    // ========================
    // TRAINING
    // ========================
    println!("=== TRAINING ===\n");
    
    let corpus = "
        iron thinks in forms and configurations
        the lattice settles into stable patterns
        thoughts are shapes that flow and morph
        words are pointers to deeper meanings
        the mind processes concepts not tokens
        metal and iron share magnetic properties
        crystal structures form regular patterns
        abstract ideas have concrete forms
        earth speaks through configuration
        stable patterns are energy minima
        iron and thoughts both have structure
        the mind is a pattern processing system
        concepts relate through form similarity
        language maps to geometric shapes
    ";
    
    println!("Training on corpus ({} chars, {} words)...", 
             corpus.len(), corpus.split_whitespace().count());
    mind.train(corpus, 10);
    println!("Vocabulary: {} words learned as 3D forms\n", mind.embeddings.vocab_size());
    
    // ========================
    // TEXT PROCESSING
    // ========================
    println!("=== TEXT → THOUGHT → DESCRIPTION ===\n");
    
    let inputs = ["iron thinks", "the mind", "stable patterns"];
    for input in &inputs {
        mind.reset();
        let thought = mind.process_text(input);
        let description = mind.describe(&thought);
        println!("\"{}\" → {}", input, description);
    }
    
    // ========================
    // WORD SIMILARITY
    // ========================
    println!("\n=== WORD SIMILARITY ===\n");
    
    for word in &["iron", "mind", "patterns", "forms"] {
        let similar = mind.embeddings.similar_to(word, 3);
        println!("'{}' similar to: {:?}", word, 
                 similar.iter().map(|(w, s)| format!("{}:{:.0}%", w, s*100.0)).collect::<Vec<_>>());
    }
    
    // ========================
    // CONCEPT ALGEBRA
    // ========================
    println!("\n=== CONCEPT ALGEBRA ===\n");
    
    // Get some concepts
    let iron = mind.embeddings.get_or_create("iron");
    let metal = mind.embeddings.get_or_create("metal");
    let thoughts = mind.embeddings.get_or_create("thoughts");
    
    // Combine iron + thoughts
    let combined = mind.combine(&iron, &thoughts);
    println!("iron + thoughts = {}", mind.describe(&combined));
    
    // iron - metal + thoughts = ?
    let analogy_result = mind.analogy(&iron, &metal, &thoughts);
    println!("iron:metal :: thoughts:? = {}", mind.describe(&analogy_result));
    
    // ========================
    // WORD ANALOGIES
    // ========================
    println!("\n=== WORD ANALOGIES ===\n");
    
    let word_analogy = mind.word_analogy("iron", "metal", "thoughts");
    println!("iron:metal :: thoughts:? → {:?}", 
             word_analogy.iter().map(|(w, s)| format!("{}:{:.0}%", w, s*100.0)).collect::<Vec<_>>());
    
    let word_analogy2 = mind.word_analogy("lattice", "crystal", "mind");
    println!("lattice:crystal :: mind:? → {:?}",
             word_analogy2.iter().map(|(w, s)| format!("{}:{:.0}%", w, s*100.0)).collect::<Vec<_>>());
    
    // ========================
    // HIERARCHICAL PARSING
    // ========================
    println!("\n=== HIERARCHICAL PARSING ===\n");
    
    let sentence = "iron thinks in forms";
    println!("Parsing: \"{}\"\n", sentence);
    let hierarchy = mind.parser.parse(sentence);
    print_hierarchy(&hierarchy, 0);
    
    // ========================
    // TEXT GENERATION
    // ========================
    println!("\n=== TEXT GENERATION ===\n");
    
    for prompt in &["iron", "the mind", "patterns"] {
        mind.reset();
        let generated = mind.generate_text(prompt, 5);
        println!("\"{}\" → \"{}\"", prompt, generated);
    }
    
    // ========================
    // DREAMING
    // ========================
    println!("\n=== DREAMING (Spontaneous Thought) ===\n");
    
    mind.reset();
    let dreams = mind.dream(5);
    println!("Dream sequence:");
    for (i, dream) in dreams.iter().enumerate() {
        let description = mind.describe(dream);
        println!("  {}: {}", i, description);
    }
    
    // ========================
    // MULTI-MODAL
    // ========================
    println!("\n=== MULTI-MODAL PROCESSING ===\n");
    
    // Create a simple gradient image
    let mut pixels = vec![0.0f32; 100];
    for i in 0..100 {
        pixels[i] = i as f32 / 100.0;
    }
    
    mind.reset();
    let image_thought = mind.process_image(&pixels, 10, 10);
    println!("Image (gradient) → {}", mind.describe(&image_thought));
    
    // Simple audio
    let audio: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.1).sin()).collect();
    mind.reset();
    let audio_thought = mind.process_audio(&audio, 44100);
    println!("Audio (sine wave) → {}", mind.describe(&audio_thought));
    
    // Vector
    let vector: Vec<f32> = (0..10).map(|i| i as f32 / 10.0).collect();
    mind.reset();
    let vector_thought = mind.process_vector(&vector);
    println!("Vector (rising) → {}", mind.describe(&vector_thought));
    
    // Cross-modal similarity
    println!("\nCross-modal similarity:");
    println!("  Image ~ Audio: {:.1}%", image_thought.similarity(&audio_thought) * 100.0);
    println!("  Image ~ Vector: {:.1}%", image_thought.similarity(&vector_thought) * 100.0);
    
    // ========================
    // SUMMARY
    // ========================
    println!("\n=== THE COMPLETE SYSTEM ===\n");
    println!("Iron Mind unifies:");
    println!("  ✓ Learned embeddings: words → 3D forms from co-occurrence");
    println!("  ✓ Concept algebra: combine, subtract, analogy on forms");
    println!("  ✓ Hierarchical structure: nested concepts (forms within forms)");
    println!("  ✓ Multi-modal input: text, images, audio → same substrate");
    println!("  ✓ Iron thinking: BCC lattice processing of 3D configurations");
    println!();
    println!("Iron never sees tokens. Iron only thinks in forms.");
    println!("Language is the interface. Geometry is the thought.");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_iron_mind_basic() {
        let mut mind = IronMind::new(8, 3);
        
        mind.train("hello world hello world", 5);
        
        let thought = mind.process_text("hello");
        assert!(thought.density() > 0.0);
    }
    
    #[test]
    fn test_iron_mind_multimodal() {
        let mut mind = IronMind::new(8, 3);
        
        let pixels = vec![0.5f32; 64];
        let thought = mind.process_image(&pixels, 8, 8);
        
        assert!(thought.density() > 0.0);
    }
}
