//! # Ferrum: Iron-Structured Computation
//!
//! A computation substrate modeled on iron's Body-Centered Cubic (BCC) atomic lattice.
//! 
//! ## The Pattern
//! 
//! Iron atoms arrange themselves in BCC structure:
//! - One atom at cube center
//! - 8 atoms at cube corners
//! - Each center atom has 8 nearest neighbors
//! 
//! We mirror this in code:
//! - Computation nodes at lattice points
//! - 8-neighbor connectivity
//! - State propagates through magnetic-like spin alignment
//! - Learning through simulated annealing
//!
//! ## Earth Speaking
//! 
//! The code is not imposed on the structure.
//! The code IS the structure.

pub mod lattice;
pub mod spin;
pub mod energy;
pub mod anneal;
pub mod carbon;
pub mod hopfield;
pub mod iron_llm;
pub mod training;

// The Core: 3D Concept-Based Intelligence
pub mod concept;      // Iron's native language: 3D forms
pub mod algebra;      // Concept operations: +, -, analogy
pub mod hierarchy;    // Forms within forms: nested structure
pub mod learned;      // Data-driven word→form mappings
pub mod multimodal;   // Images, audio, vectors → 3D forms
pub mod concept_lm;   // Text → Concepts → Iron → Text
pub mod iron_mind;    // Unified system: the complete Iron Mind

pub use lattice::BCCLattice;
pub use spin::Spin;
pub use energy::Energy;
pub use hopfield::{IronMemory, Pattern};
pub use iron_llm::{IronLLM, IronLayer, Embedding, CharTokenizer};
pub use training::{IronTrainer, TrainConfig, demo_training};
pub use iron_mind::IronMind;
