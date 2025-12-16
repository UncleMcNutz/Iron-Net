//! # Copper Module: Transmission and Flow
//!
//! If Iron (BCC) is for **holding and thinking**,
//! Copper (FCC) is for **flowing and transmitting**.
//!
//! ## The Metaphor
//!
//! - Iron: Configuration, memory, state, magnetism → THOUGHT
//! - Copper: Conductivity, channels, flow, no memory → TRANSMISSION
//!
//! ## FCC vs BCC
//!
//! - BCC (Iron): 8 nearest neighbors - tight, strong bonds
//! - FCC (Copper): 12 nearest neighbors - more connections, easier flow
//!
//! ## Go as Copper's Language
//!
//! If Rust is Iron (ownership=bonding, lifetimes=crystal stability),
//! Go is Copper (channels=conductivity, goroutines=parallel flow).
//!
//! We model Copper in Rust, but with flow-based semantics.

pub mod lattice;
pub mod channel;

pub use lattice::{FCCLattice, CopperNode};
pub use channel::{ChannelNetwork, Endpoint, Message};
