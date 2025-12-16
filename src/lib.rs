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

pub use lattice::BCCLattice;
pub use spin::Spin;
pub use energy::Energy;
pub use hopfield::{IronMemory, Pattern};
pub use iron_llm::{IronLLM, IronLayer, Embedding, CharTokenizer};
