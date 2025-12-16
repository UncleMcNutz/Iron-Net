//! # Alloy: Iron + Copper Integration
//!
//! Iron holds thoughts. Copper transmits them.
//! Together, they form a thinking-communicating system.
//!
//! ## The Architecture
//!
//! ```text
//!                    ┌─────────────┐
//!                    │   ALLOY     │
//!                    │  (Fe + Cu)  │
//!                    └──────┬──────┘
//!                           │
//!          ┌────────────────┼────────────────┐
//!          │                │                │
//!          ▼                ▼                ▼
//!     ┌─────────┐     ┌─────────┐     ┌─────────┐
//!     │  IRON   │     │  IRON   │     │  IRON   │
//!     │ Region  │     │ Region  │     │ Region  │
//!     │  (BCC)  │     │  (BCC)  │     │  (BCC)  │
//!     │ THOUGHT │     │ THOUGHT │     │ THOUGHT │
//!     └────┬────┘     └────┬────┘     └────┬────┘
//!          │               │               │
//!          └───────────────┼───────────────┘
//!                          │
//!                    ┌─────▼─────┐
//!                    │  COPPER   │
//!                    │  Network  │
//!                    │   (FCC)   │
//!                    │   FLOW    │
//!                    └───────────┘
//! ```
//!
//! ## Design Principle
//!
//! - Iron regions: Independent processing units (like neurons)
//! - Copper network: Communication infrastructure (like axons)
//! - Each iron region processes concepts locally
//! - Copper routes concepts between regions

use crate::lattice::BCCLattice;
use crate::hopfield::IronMemory;
use crate::concept::Concept;
use crate::spin::Spin;
use crate::copper::ChannelNetwork;
use std::collections::HashMap;

/// A region of iron - a local processing unit
pub struct IronRegion {
    /// Unique identifier
    pub id: usize,
    /// Name
    pub name: String,
    /// The BCC lattice for this region
    pub lattice: BCCLattice,
    /// Hopfield-style memory (2D slice for pattern storage)
    pub memory: IronMemory,
    /// Current concept being processed
    pub current_concept: Option<Concept>,
    /// Copper endpoint ID for communication
    pub copper_endpoint: usize,
}

impl IronRegion {
    pub fn new(id: usize, name: &str, size: usize) -> Self {
        Self {
            id,
            name: name.to_string(),
            lattice: BCCLattice::new(size, size, size),
            memory: IronMemory::new(size, size),  // 2D memory slice
            current_concept: None,
            copper_endpoint: 0,
        }
    }
    
    /// Inject a concept for processing
    pub fn inject(&mut self, concept: Concept) {
        // Load concept into lattice
        concept.into_lattice(&mut self.lattice);
        self.current_concept = Some(concept);
    }
    
    /// Process one step (anneal) - let the iron think
    pub fn process(&mut self, temperature: f32) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // Pick random cell and potentially flip
        let idx = rng.gen_range(0..self.lattice.cells.len());
        let energy_before = self.local_energy(idx);
        
        // Flip
        self.lattice.cells[idx].spin = self.lattice.cells[idx].spin.flipped();
        
        let energy_after = self.local_energy(idx);
        let delta_e = energy_after - energy_before;
        
        // Accept or reject based on Metropolis criterion
        if delta_e > 0.0 {
            let accept_prob = (-delta_e / temperature.max(0.001)).exp();
            if rng.gen::<f32>() > accept_prob {
                // Reject - flip back
                self.lattice.cells[idx].spin = self.lattice.cells[idx].spin.flipped();
            }
        }
        // If delta_e <= 0, always accept (lower energy is good)
    }
    
    /// Calculate local energy around a cell
    fn local_energy(&self, idx: usize) -> f32 {
        let cell = &self.lattice.cells[idx];
        let spin_val = cell.spin.value();
        
        let mut energy = 0.0;
        for (i, &neighbor_idx) in cell.neighbors.iter().enumerate() {
            let neighbor_spin = self.lattice.cells[neighbor_idx].spin.value();
            energy -= cell.coupling[i] * spin_val * neighbor_spin;
        }
        energy
    }
    
    /// Extract current state as concept
    pub fn extract(&self) -> Option<Concept> {
        if self.current_concept.is_some() {
            // Return the current state as a new concept
            Some(Concept::from_lattice(&self.lattice))
        } else {
            None
        }
    }
    
    /// Store current concept in memory
    pub fn memorize(&mut self) {
        if let Some(concept) = &self.current_concept {
            // Convert concept form to pattern
            let pattern = crate::hopfield::Pattern {
                spins: concept.form.clone(),
                label: concept.label.clone(),
            };
            self.memory.store(&pattern);
        }
    }
    
    /// Recall from memory (pattern completion)
    /// Uses current concept as probe to recall similar stored pattern
    pub fn recall(&mut self, iterations: usize) {
        if let Some(concept) = &self.current_concept {
            // Create probe pattern from current concept
            let probe = crate::hopfield::Pattern {
                spins: concept.form.clone(),
                label: concept.label.clone(),
            };
            // Recall from memory
            let recalled = self.memory.recall(&probe, iterations);
            // Update current concept with recalled pattern
            let dims = self.current_concept.as_ref().map(|c| c.dims).unwrap_or((8, 8, 8));
            self.current_concept = Some(Concept {
                form: recalled.spins,
                dims,
                label: recalled.label,
            });
        }
    }
}

/// The alloy system - Iron regions connected by Copper
pub struct Alloy {
    /// Iron regions (processing units)
    pub regions: Vec<IronRegion>,
    /// Copper network (communication)
    pub copper: ChannelNetwork,
    /// Region name to ID mapping
    pub region_map: HashMap<String, usize>,
}

impl Alloy {
    pub fn new(copper_size: usize) -> Self {
        Self {
            regions: Vec::new(),
            copper: ChannelNetwork::new(copper_size),
            region_map: HashMap::new(),
        }
    }
    
    /// Add a new iron region
    pub fn add_region(&mut self, name: &str, size: usize) -> usize {
        let region_id = self.regions.len();
        
        // Register with copper network
        let endpoint_id = self.copper.register(name);
        
        let mut region = IronRegion::new(region_id, name, size);
        region.copper_endpoint = endpoint_id;
        
        self.region_map.insert(name.to_string(), region_id);
        self.regions.push(region);
        
        region_id
    }
    
    /// Add region at specific copper position
    pub fn add_region_at(&mut self, name: &str, size: usize, cx: usize, cy: usize, cz: usize) -> usize {
        let region_id = self.regions.len();
        
        // Register with copper network at position
        let endpoint_id = self.copper.register_at(name, cx, cy, cz);
        
        let mut region = IronRegion::new(region_id, name, size);
        region.copper_endpoint = endpoint_id;
        
        self.region_map.insert(name.to_string(), region_id);
        self.regions.push(region);
        
        region_id
    }
    
    /// Get region by name
    pub fn region(&self, name: &str) -> Option<&IronRegion> {
        self.region_map.get(name).map(|&id| &self.regions[id])
    }
    
    /// Get mutable region by name
    pub fn region_mut(&mut self, name: &str) -> Option<&mut IronRegion> {
        if let Some(&id) = self.region_map.get(name) {
            Some(&mut self.regions[id])
        } else {
            None
        }
    }
    
    /// Send concept from one region to another
    pub fn transmit(&mut self, from: &str, to: &str) {
        let (from_endpoint, concept) = {
            if let Some(region) = self.region(from) {
                (region.copper_endpoint, region.extract())
            } else {
                return;
            }
        };
        
        let to_endpoint = {
            if let Some(region) = self.region(to) {
                region.copper_endpoint
            } else {
                return;
            }
        };
        
        if let Some(concept) = concept {
            // Serialize concept to payload
            let payload = concept_to_payload(&concept);
            
            // Send through copper
            if let Some(ep) = self.copper.endpoint_mut(from_endpoint) {
                ep.send(payload, to_endpoint);
            }
        }
    }
    
    /// Broadcast concept from one region to all others
    pub fn broadcast(&mut self, from: &str) {
        let (from_endpoint, concept) = {
            if let Some(region) = self.region(from) {
                (region.copper_endpoint, region.extract())
            } else {
                return;
            }
        };
        
        if let Some(concept) = concept {
            let payload = concept_to_payload(&concept);
            
            if let Some(ep) = self.copper.endpoint_mut(from_endpoint) {
                ep.broadcast(payload);
            }
        }
    }
    
    /// Route messages through copper
    pub fn route(&mut self, steps: usize) {
        for _ in 0..steps {
            self.copper.route();
        }
        
        // Deliver received messages to iron regions
        for region in &mut self.regions {
            let endpoint_id = region.copper_endpoint;
            if let Some(ep) = self.copper.endpoint_mut(endpoint_id) {
                while let Some(msg) = ep.receive() {
                    // Deserialize concept from payload
                    if let Some(concept) = payload_to_concept(&msg.payload, region.lattice.size_x) {
                        region.inject(concept);
                    }
                }
            }
        }
    }
    
    /// Process all regions for one step
    pub fn think(&mut self, temperature: f32) {
        for region in &mut self.regions {
            // Multiple micro-steps per think step
            for _ in 0..100 {
                region.process(temperature);
            }
        }
    }
    
    /// Full cycle: think, then route
    pub fn cycle(&mut self, think_steps: usize, route_steps: usize, temperature: f32) {
        for _ in 0..think_steps {
            self.think(temperature);
        }
        self.route(route_steps);
    }
}

/// Convert concept to wire format
fn concept_to_payload(concept: &Concept) -> Vec<f32> {
    let mut payload = Vec::new();
    
    // Header: dimensions
    payload.push(concept.dims.0 as f32);
    payload.push(concept.dims.1 as f32);
    payload.push(concept.dims.2 as f32);
    
    // Spins as floats
    for spin in &concept.form {
        payload.push(spin.value());
    }
    
    payload
}

/// Convert wire format to concept
fn payload_to_concept(payload: &[f32], default_size: usize) -> Option<Concept> {
    if payload.len() < 3 {
        return None;
    }
    
    let dims = (
        payload[0] as usize,
        payload[1] as usize,
        payload[2] as usize,
    );
    
    // Extract spins
    let form: Vec<Spin> = payload[3..].iter()
        .map(|&v| if v > 0.0 { Spin::Up } else { Spin::Down })
        .collect();
    
    // Ensure correct size
    let expected = dims.0 * dims.1 * dims.2;
    let form = if form.len() >= expected {
        form[..expected].to_vec()
    } else {
        // Pad with Down spins
        let mut padded = form;
        padded.resize(expected.max(default_size.pow(3)), Spin::Down);
        padded
    };
    
    Some(Concept {
        form,
        dims,
        label: Some("received".to_string()),
    })
}

/// Demo the alloy system
pub fn demo_alloy() {
    println!("=== ALLOY: Iron + Copper Integration ===\n");
    
    let mut alloy = Alloy::new(16);
    
    // Create regions
    let _perception = alloy.add_region_at("perception", 8, 0, 0, 0);
    let _reasoning = alloy.add_region_at("reasoning", 8, 8, 8, 8);
    let _language = alloy.add_region_at("language", 8, 15, 15, 15);
    
    println!("Created regions:");
    for region in &alloy.regions {
        println!("  {} (endpoint {})", region.name, region.copper_endpoint);
    }
    
    // Create and inject a concept into perception
    let dog_concept = Concept::empty(8, 8, 8).with_label("dog");
    
    // Sculpt the dog concept - add some structure
    let mut concept = dog_concept;
    // Center mass (body)
    for z in 3..5 {
        for y in 3..5 {
            for x in 2..6 {
                let idx = x + y * 8 + z * 64;
                if idx < concept.form.len() {
                    concept.form[idx] = Spin::Up;
                }
            }
        }
    }
    
    if let Some(region) = alloy.region_mut("perception") {
        region.inject(concept);
        println!("\nInjected 'dog' concept into perception");
    }
    
    // Process perception
    println!("\nProcessing perception (iron thinks)...");
    alloy.think(0.1);
    
    // Transmit from perception to reasoning
    println!("Transmitting from perception to reasoning (copper flows)...");
    alloy.transmit("perception", "reasoning");
    
    // Route through copper
    alloy.route(20);
    
    // Check if reasoning received
    if let Some(region) = alloy.region("reasoning") {
        if region.current_concept.is_some() {
            println!("✓ Reasoning received concept!");
        }
    }
    
    // Process reasoning
    println!("\nProcessing reasoning (iron thinks about received concept)...");
    alloy.think(0.1);
    
    // Broadcast from reasoning to all
    println!("Broadcasting from reasoning (copper spreads the thought)...");
    alloy.broadcast("reasoning");
    alloy.route(20);
    
    // Check language
    if let Some(region) = alloy.region("language") {
        if region.current_concept.is_some() {
            println!("✓ Language received concept!");
        }
    }
    
    println!("\n=== Iron thinks, Copper transmits ===");
    println!("=== Together: the Alloy mind ===");
}

// Helper trait for Concept
trait ConceptExt {
    fn with_label(self, label: &str) -> Self;
}

impl ConceptExt for Concept {
    fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_alloy_transmission() {
        let mut alloy = Alloy::new(8);
        alloy.add_region_at("a", 4, 0, 0, 0);
        alloy.add_region_at("b", 4, 7, 7, 7);
        
        let concept = Concept::empty(4, 4, 4).with_label("test");
        
        alloy.region_mut("a").unwrap().inject(concept);
        alloy.transmit("a", "b");
        alloy.route(20);
        
        assert!(alloy.region("b").unwrap().current_concept.is_some());
    }
}
