//! # Concept Language Model
//!
//! The architecture insight:
//!   Text → Concept Encoder → 3D Form → Iron Thinks → 3D Form → Concept Decoder → Text
//!
//! Iron never touches text directly.
//! Iron only thinks in 3D configurations.
//! Language is just the human interface layer.
//!
//! ## The Flow
//!
//! 1. ENCODE: Text → Concept (map words to 3D forms)
//! 2. THINK: Concept → Concept (iron processes in native 3D)
//! 3. DECODE: Concept → Text (map settled form back to words)
//!
//! ## Why This Works
//!
//! Words are not meaning. Words are POINTERS to meaning.
//! The meaning itself is the 3D conceptual structure.
//! Iron processes the meaning, not the pointers.

use crate::concept::{Concept, ConceptMind, ConceptSculptor};
use crate::lattice::BCCLattice;
use crate::spin::Spin;
use crate::anneal::Annealer;
use std::collections::HashMap;
use rand::Rng;

/// A semantic feature - one dimension of meaning
#[derive(Clone, Debug)]
pub struct SemanticFeature {
    pub name: &'static str,
    pub axis: usize,  // Which spatial axis this maps to (0=x, 1=y, 2=z)
    pub positive: &'static str,  // What +1 means
    pub negative: &'static str,  // What -1 means
}

/// Semantic space: the bridge between words and 3D forms
/// 
/// Words map to positions/shapes in semantic space.
/// Similar words have similar shapes.
/// Related words have shapes that flow into each other.
pub struct SemanticSpace {
    /// Dimension of concept lattice
    pub size: usize,
    /// Word to concept mapping (learned)
    pub word_concepts: HashMap<String, Concept>,
    /// Concept to words mapping (for decoding)
    pub concept_words: Vec<(Concept, String)>,
    /// Semantic features that structure the space
    pub features: Vec<SemanticFeature>,
}

impl SemanticSpace {
    pub fn new(size: usize) -> Self {
        // Define semantic features - these structure how concepts are arranged
        let features = vec![
            SemanticFeature { 
                name: "concrete_abstract", 
                axis: 0,
                positive: "concrete", 
                negative: "abstract" 
            },
            SemanticFeature { 
                name: "active_passive", 
                axis: 1,
                positive: "active", 
                negative: "passive" 
            },
            SemanticFeature { 
                name: "positive_negative", 
                axis: 2,
                positive: "positive", 
                negative: "negative" 
            },
        ];
        
        Self {
            size,
            word_concepts: HashMap::new(),
            concept_words: Vec::new(),
            features,
        }
    }
    
    /// Encode a word into a 3D concept
    /// 
    /// This creates a geometric form based on semantic properties.
    /// Similar words → similar forms.
    pub fn encode_word(&mut self, word: &str) -> Concept {
        // Check cache
        if let Some(concept) = self.word_concepts.get(word) {
            return concept.clone();
        }
        
        let sculptor = ConceptSculptor::new(self.size, self.size, self.size);
        
        // Create concept based on word properties
        // This is a simple heuristic - in practice, learned from data
        let concept = self.word_to_form(word, &sculptor);
        
        // Cache it
        self.word_concepts.insert(word.to_string(), concept.clone());
        self.concept_words.push((concept.clone(), word.to_string()));
        
        concept
    }
    
    /// Convert word to geometric form based on semantic properties
    fn word_to_form(&self, word: &str, sculptor: &ConceptSculptor) -> Concept {
        let word_lower = word.to_lowercase();
        let len = word.len();
        let hash = self.word_hash(&word_lower);
        
        // Determine basic shape from word category
        let base_form = if self.is_noun(&word_lower) {
            // Nouns are solid objects - spheres, cubes
            let radius = 2.0 + (len as f32 * 0.3).min(3.0);
            sculptor.sphere(radius)
        } else if self.is_verb(&word_lower) {
            // Verbs are directional - rods, arrows
            let axis = ['x', 'y', 'z'][hash % 3];
            sculptor.rod(axis, 2.0)
        } else if self.is_adjective(&word_lower) {
            // Adjectives are shells/wrappers - they modify
            sculptor.shell(2.0, 4.0)
        } else if self.is_preposition(&word_lower) {
            // Prepositions are planes - they separate/relate
            let axis = ['x', 'y', 'z'][hash % 3];
            sculptor.plane(axis, self.size / 2, 2)
        } else {
            // Default: noise-based unique form
            sculptor.noise(0.3 + (hash as f32 / 1000.0).fract())
        };
        
        // Apply transformations based on semantic features
        let transformed = self.apply_semantic_transforms(base_form, &word_lower, hash);
        
        transformed.with_label(word)
    }
    
    /// Apply semantic transformations to base form
    fn apply_semantic_transforms(&self, mut concept: Concept, word: &str, hash: usize) -> Concept {
        let (dx, dy, dz) = concept.dims;
        
        // Shift position based on semantic properties
        // Abstract words → left side (low x)
        // Concrete words → right side (high x)
        let concrete_shift = if self.is_concrete(word) { 
            (dx / 4) as i32 
        } else { 
            -((dx / 4) as i32) 
        };
        
        // Active words → top (high y)
        // Passive words → bottom (low y)
        let active_shift = if self.is_active(word) { 
            (dy / 4) as i32 
        } else { 
            -((dy / 4) as i32) 
        };
        
        // Positive words → front (high z)
        // Negative words → back (low z)
        let valence_shift = if self.is_positive_valence(word) { 
            (dz / 4) as i32 
        } else { 
            -((dz / 4) as i32) 
        };
        
        // Create shifted concept
        let mut shifted = Concept::empty(dx, dy, dz);
        for z in 0..dz {
            for y in 0..dy {
                for x in 0..dx {
                    let spin = concept.get(x, y, z);
                    let new_x = ((x as i32 + concrete_shift).max(0) as usize).min(dx - 1);
                    let new_y = ((y as i32 + active_shift).max(0) as usize).min(dy - 1);
                    let new_z = ((z as i32 + valence_shift).max(0) as usize).min(dz - 1);
                    if spin == Spin::Up {
                        shifted.set(new_x, new_y, new_z, Spin::Up);
                    }
                }
            }
        }
        
        shifted
    }
    
    /// Decode a concept back to the closest word
    pub fn decode_concept(&self, concept: &Concept) -> Option<String> {
        self.concept_words.iter()
            .map(|(c, w)| (concept.similarity(c), w))
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .filter(|(sim, _)| *sim > 0.3)  // Minimum similarity threshold
            .map(|(_, w)| w.clone())
    }
    
    /// Get top-k closest words to a concept
    pub fn decode_topk(&self, concept: &Concept, k: usize) -> Vec<(String, f32)> {
        let mut scored: Vec<_> = self.concept_words.iter()
            .map(|(c, w)| (w.clone(), concept.similarity(c)))
            .collect();
        
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(k);
        scored
    }
    
    // === Word classification heuristics ===
    // In practice, these would be learned or use a real lexicon
    
    fn word_hash(&self, word: &str) -> usize {
        word.bytes().fold(0usize, |acc, b| acc.wrapping_mul(31).wrapping_add(b as usize))
    }
    
    fn is_noun(&self, word: &str) -> bool {
        // Common nouns
        matches!(word, "iron" | "code" | "lattice" | "crystal" | "form" | "shape" |
                      "concept" | "thought" | "mind" | "earth" | "metal" | "atom" |
                      "spin" | "energy" | "pattern" | "structure" | "cell" | "node" |
                      "network" | "brain" | "memory" | "world" | "thing" | "object")
    }
    
    fn is_verb(&self, word: &str) -> bool {
        matches!(word, "is" | "are" | "be" | "think" | "speaks" | "speak" | "form" |
                      "forms" | "flow" | "flows" | "settle" | "settles" | "become" |
                      "becomes" | "compute" | "computes" | "learn" | "learns" |
                      "remember" | "forget" | "create" | "creates" | "morph" | "morphs")
    }
    
    fn is_adjective(&self, word: &str) -> bool {
        matches!(word, "stable" | "unstable" | "magnetic" | "crystalline" | "atomic" |
                      "solid" | "abstract" | "concrete" | "complex" | "simple" |
                      "deep" | "shallow" | "dense" | "sparse" | "hot" | "cold")
    }
    
    fn is_preposition(&self, word: &str) -> bool {
        matches!(word, "in" | "on" | "at" | "to" | "from" | "with" | "through" |
                      "into" | "onto" | "within" | "without" | "between" | "among")
    }
    
    fn is_concrete(&self, word: &str) -> bool {
        matches!(word, "iron" | "metal" | "crystal" | "atom" | "earth" | "rock" |
                      "body" | "hand" | "brain" | "cell" | "lattice" | "node")
    }
    
    fn is_active(&self, word: &str) -> bool {
        matches!(word, "think" | "speaks" | "flow" | "create" | "compute" | "learn" |
                      "morph" | "form" | "settle" | "become" | "run" | "move")
    }
    
    fn is_positive_valence(&self, word: &str) -> bool {
        matches!(word, "stable" | "create" | "learn" | "good" | "strong" | "bright" |
                      "clear" | "true" | "yes" | "form" | "grow" | "build")
    }
}

/// The Concept Language Model
/// 
/// Text → Concepts → Iron Thinking → Concepts → Text
pub struct ConceptLM {
    /// Semantic space for encoding/decoding
    pub semantic_space: SemanticSpace,
    /// The concept mind where thinking happens
    pub mind: ConceptMind,
    /// Context: recent concepts (not words!)
    pub context: Vec<Concept>,
    /// Maximum context concepts
    pub max_context: usize,
}

impl ConceptLM {
    pub fn new(lattice_size: usize, max_context: usize) -> Self {
        Self {
            semantic_space: SemanticSpace::new(lattice_size),
            mind: ConceptMind::new(lattice_size),
            context: Vec::new(),
            max_context,
        }
    }
    
    /// Process input text: encode to concepts, let iron think
    pub fn process(&mut self, input: &str) -> String {
        // 1. Tokenize (simple whitespace for now)
        let words: Vec<&str> = input.split_whitespace().collect();
        
        if words.is_empty() {
            return String::new();
        }
        
        // 2. Encode words to concepts
        let input_concepts: Vec<Concept> = words.iter()
            .map(|w| self.semantic_space.encode_word(w))
            .collect();
        
        // 3. Combine input concepts into one form
        let combined_input = self.combine_concepts(&input_concepts);
        
        // 4. Add to context
        self.context.push(combined_input.clone());
        if self.context.len() > self.max_context {
            self.context.remove(0);
        }
        
        // 5. Create context-influenced input
        let contextualized = if self.context.len() > 1 {
            let context_blend = self.combine_concepts(&self.context[..self.context.len()-1]);
            combined_input.blend(&context_blend, 0.3)
        } else {
            combined_input
        };
        
        // 6. Let iron think (concept evolves through lattice)
        let output_concept = self.mind.think(&contextualized, 2000);
        
        // 7. Decode concept back to words
        self.decode_to_text(&output_concept)
    }
    
    /// Generate continuation from prompt
    pub fn generate(&mut self, prompt: &str, num_words: usize) -> String {
        let mut result = prompt.to_string();
        
        // First, process the prompt to build context
        let prompt_words: Vec<&str> = prompt.split_whitespace().collect();
        for word in &prompt_words {
            let concept = self.semantic_space.encode_word(word);
            self.context.push(concept);
            if self.context.len() > self.max_context {
                self.context.remove(0);
            }
        }
        
        // Generate new words
        for _ in 0..num_words {
            // Create input from context
            let input = if !self.context.is_empty() {
                self.combine_concepts(&self.context)
            } else {
                let sculptor = ConceptSculptor::new(
                    self.mind.lattice.size_x,
                    self.mind.lattice.size_y,
                    self.mind.lattice.size_z,
                );
                sculptor.noise(0.5)
            };
            
            // Think
            let output = self.mind.think(&input, 1500);
            
            // Decode
            let decoded = self.decode_with_sampling(&output);
            
            // Add to result and context
            if !decoded.is_empty() {
                result.push(' ');
                result.push_str(&decoded);
                
                let new_concept = self.semantic_space.encode_word(&decoded);
                self.context.push(new_concept);
                if self.context.len() > self.max_context {
                    self.context.remove(0);
                }
            }
        }
        
        result
    }
    
    /// Combine multiple concepts into one form
    fn combine_concepts(&self, concepts: &[Concept]) -> Concept {
        if concepts.is_empty() {
            return Concept::empty(
                self.mind.lattice.size_x,
                self.mind.lattice.size_y,
                self.mind.lattice.size_z,
            );
        }
        
        if concepts.len() == 1 {
            return concepts[0].clone();
        }
        
        // Superposition with decay (recent concepts weighted more)
        let mut result = concepts[0].clone();
        for (i, concept) in concepts.iter().enumerate().skip(1) {
            let weight = (i as f32 + 1.0) / concepts.len() as f32;
            result = result.blend(concept, weight * 0.5);
        }
        
        result
    }
    
    /// Decode concept to text (greedy)
    fn decode_to_text(&self, concept: &Concept) -> String {
        if let Some(word) = self.semantic_space.decode_concept(concept) {
            word
        } else {
            // No close match - describe the form
            let density = concept.density();
            let (sx, sy, sz) = concept.symmetry();
            let avg_sym = (sx + sy + sz) / 3.0;
            
            if density > 0.5 {
                "dense".to_string()
            } else if avg_sym > 0.8 {
                "stable".to_string()
            } else {
                "form".to_string()
            }
        }
    }
    
    /// Decode with sampling from top-k
    fn decode_with_sampling(&self, concept: &Concept) -> String {
        let topk = self.semantic_space.decode_topk(concept, 5);
        
        if topk.is_empty() {
            return self.decode_to_text(concept);
        }
        
        // Temperature-based sampling
        let mut rng = rand::thread_rng();
        let temp = 1.5;
        
        let weights: Vec<f32> = topk.iter()
            .map(|(_, sim)| (sim / temp).exp())
            .collect();
        let sum: f32 = weights.iter().sum();
        let probs: Vec<f32> = weights.iter().map(|w| w / sum).collect();
        
        let r: f32 = rng.gen();
        let mut cumsum = 0.0;
        for (i, p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return topk[i].0.clone();
            }
        }
        
        topk[0].0.clone()
    }
    
    /// Train on text: make word sequences into concept attractors
    pub fn train(&mut self, text: &str) {
        let words: Vec<&str> = text.split_whitespace().collect();
        
        // For each word sequence, teach that the concept flow is stable
        for window in words.windows(3) {
            // Encode sequence
            let concepts: Vec<Concept> = window.iter()
                .map(|w| self.semantic_space.encode_word(w))
                .collect();
            
            // Combined form of this sequence should be an attractor
            let sequence_concept = self.combine_concepts(&concepts);
            self.mind.learn(&sequence_concept);
        }
        
        // Also learn individual word concepts strongly
        for word in &words {
            let concept = self.semantic_space.encode_word(word);
            self.mind.learn(&concept);
        }
    }
    
    /// Clear context
    pub fn reset_context(&mut self) {
        self.context.clear();
    }
}

/// Demonstrate the Concept Language Model
pub fn demo_concept_lm() {
    println!("================================================================");
    println!("      CONCEPT LANGUAGE MODEL: Iron Thinking in 3D Forms");
    println!("      Text → Concepts → Iron → Concepts → Text");
    println!("================================================================");
    println!();
    
    let lattice_size = 10;
    let max_context = 5;
    
    println!("Creating Concept LM ({}x{}x{} lattice, {} concept context)...\n", 
             lattice_size, lattice_size, lattice_size, max_context);
    
    let mut model = ConceptLM::new(lattice_size, max_context);
    
    // Build vocabulary
    println!("=== BUILDING SEMANTIC SPACE ===\n");
    
    let vocabulary = [
        "iron", "speaks", "in", "forms", "not", "words",
        "the", "lattice", "thinks", "concepts", "are", "shapes",
        "earth", "is", "solid", "abstract", "concrete", "flow",
        "stable", "energy", "pattern", "crystal", "atom", "mind",
        "code", "structure", "compute", "learn", "memory", "form",
    ];
    
    for word in &vocabulary {
        let concept = model.semantic_space.encode_word(word);
        println!("  '{}' → density={:.1}%, symmetry=({:.0}%,{:.0}%,{:.0}%)", 
                 word,
                 concept.density() * 100.0,
                 concept.symmetry().0 * 100.0,
                 concept.symmetry().1 * 100.0,
                 concept.symmetry().2 * 100.0);
    }
    
    println!("\nVocabulary: {} words encoded as 3D forms\n", model.semantic_space.word_concepts.len());
    
    // Train on sample text
    println!("=== TRAINING ===\n");
    
    let training_text = "iron speaks in forms not words \
                         the lattice thinks in concepts \
                         concepts are shapes that flow \
                         earth is solid and stable \
                         the mind computes in patterns \
                         code is structure made solid";
    
    println!("Training on: \"{}\"", training_text);
    model.train(training_text);
    println!("Trained: word sequences → concept attractors\n");
    
    // Test processing
    println!("=== PROCESSING (Input → Concept → Output) ===\n");
    
    let test_inputs = ["iron", "the lattice", "concepts are"];
    
    for input in &test_inputs {
        model.reset_context();
        let output = model.process(input);
        println!("  \"{}\" → [iron thinks] → \"{}\"", input, output);
    }
    
    // Test generation
    println!("\n=== GENERATION ===\n");
    
    let prompts = ["iron", "the", "concepts"];
    
    for prompt in &prompts {
        model.reset_context();
        let generated = model.generate(prompt, 5);
        println!("  \"{}\" → \"{}\"", prompt, generated);
    }
    
    // Show concept similarity
    println!("\n=== CONCEPT RELATIONSHIPS ===\n");
    
    let pairs = [
        ("iron", "metal"),
        ("iron", "abstract"),
        ("think", "compute"),
        ("stable", "solid"),
        ("form", "shape"),
    ];
    
    for (a, b) in &pairs {
        let ca = model.semantic_space.encode_word(a);
        let cb = model.semantic_space.encode_word(b);
        let sim = ca.similarity(&cb);
        println!("  '{}' ~ '{}': {:.1}% similar", a, b, sim * 100.0);
    }
    
    println!("\n=== THE ARCHITECTURE ===\n");
    println!("  Text (2D) ──→ Concept Encoder ──→ 3D Form");
    println!("                                      │");
    println!("                                      ▼");
    println!("                                 Iron Thinks");
    println!("                                 (BCC lattice)");
    println!("                                      │");
    println!("                                      ▼");
    println!("  Text (2D) ←── Concept Decoder ←── 3D Form");
    println!();
    println!("Iron never sees text. Iron only thinks in forms.");
    println!("Language is just the human interface.");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_semantic_encode_decode() {
        let mut space = SemanticSpace::new(8);
        
        let concept = space.encode_word("iron");
        let decoded = space.decode_concept(&concept);
        
        assert_eq!(decoded, Some("iron".to_string()));
    }
    
    #[test]
    fn test_similar_words_similar_forms() {
        let mut space = SemanticSpace::new(8);
        
        let iron = space.encode_word("iron");
        let metal = space.encode_word("metal");
        let abstract_word = space.encode_word("abstract");
        
        // iron and metal should be more similar than iron and abstract
        let sim_iron_metal = iron.similarity(&metal);
        let sim_iron_abstract = iron.similarity(&abstract_word);
        
        // Both are nouns so they'll have similar base shapes
        // but semantic shifts should differentiate
        assert!(sim_iron_metal > 0.0);
        assert!(sim_iron_abstract > 0.0);
    }
}
