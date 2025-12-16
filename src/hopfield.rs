//! # Hopfield Network: Iron's Natural Computation
//!
//! A Hopfield network IS an Ising model with learned couplings.
//! This is what iron-structured computation does naturally:
//! 
//! - Store patterns as energy minima
//! - Recall complete patterns from partial/noisy input
//! - Associative memory: content-addressable, not location-addressable
//!
//! The BCC lattice becomes a memory that "wants" to be in certain states.
//! Give it a hint, and it falls into the nearest stored pattern.
//!
//! This is fundamentally different from LLMs:
//! - LLMs: predict next token sequentially
//! - Hopfield: recall entire pattern simultaneously
//!
//! Iron doesn't think in sequences. Iron thinks in stable configurations.

use crate::lattice::BCCLattice;
use crate::spin::Spin;
use crate::anneal::Annealer;

/// A pattern that can be stored in the iron memory
#[derive(Clone, Debug)]
pub struct Pattern {
    /// The spin configuration (flattened)
    pub spins: Vec<Spin>,
    /// Optional label for the pattern
    pub label: Option<String>,
}

impl Pattern {
    /// Create a pattern from a slice of booleans
    pub fn from_bools(data: &[bool], label: Option<&str>) -> Self {
        Self {
            spins: data.iter().map(|&b| Spin::from_bool(b)).collect(),
            label: label.map(String::from),
        }
    }
    
    /// Create a pattern from a 2D grid (for visualization)
    pub fn from_grid(grid: &[&[bool]], label: Option<&str>) -> Self {
        let spins: Vec<Spin> = grid.iter()
            .flat_map(|row| row.iter().map(|&b| Spin::from_bool(b)))
            .collect();
        Self {
            spins,
            label: label.map(String::from),
        }
    }
    
    /// Create pattern from string (# = up, . = down)
    pub fn from_string(s: &str, label: Option<&str>) -> Self {
        let spins: Vec<Spin> = s.chars()
            .filter(|c| *c == '#' || *c == '.')
            .map(|c| Spin::from_bool(c == '#'))
            .collect();
        Self {
            spins,
            label: label.map(String::from),
        }
    }
    
    /// Corrupt the pattern with noise (flip random bits)
    pub fn corrupt(&self, noise_fraction: f32) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let spins = self.spins.iter()
            .map(|&s| {
                if rng.gen::<f32>() < noise_fraction {
                    s.flipped()
                } else {
                    s
                }
            })
            .collect();
        Self {
            spins,
            label: self.label.clone(),
        }
    }
    
    /// Compute overlap with another pattern (-1 to +1)
    pub fn overlap(&self, other: &Pattern) -> f32 {
        let n = self.spins.len().min(other.spins.len());
        if n == 0 { return 0.0; }
        
        let sum: f32 = self.spins.iter()
            .zip(other.spins.iter())
            .map(|(a, b)| a.value() * b.value())
            .sum();
        sum / n as f32
    }
}

/// Hopfield-style associative memory using BCC lattice
pub struct IronMemory {
    /// The underlying lattice (2D for visualization, but uses BCC connectivity)
    pub lattice: BCCLattice,
    /// Stored patterns
    pub patterns: Vec<Pattern>,
    /// Size of patterns (width * height for 2D)
    pub pattern_size: usize,
}

impl IronMemory {
    /// Create a new iron memory with given dimensions
    /// Uses a thin lattice (z=1) for 2D pattern storage
    pub fn new(width: usize, height: usize) -> Self {
        // Use z=2 to still get some 3D BCC character
        let mut lattice = BCCLattice::new(width, height, 2);
        
        // Start with ZERO couplings - patterns will define the structure
        // This is pure associative memory, not the BCC ground state
        for cell in &mut lattice.cells {
            cell.coupling = [0.0; 8];
        }
        
        Self {
            lattice,
            patterns: Vec::new(),
            pattern_size: width * height,
        }
    }
    
    /// Store a pattern using Hebbian learning
    /// 
    /// The Hebb rule: "neurons that fire together, wire together"
    /// J_ij += (1/N) * s_i * s_j for each stored pattern
    /// 
    /// This makes the stored patterns into energy minima
    pub fn store(&mut self, pattern: &Pattern) {
        let n = self.pattern_size;
        let num_cells = self.lattice.cells.len();
        
        // Collect updates first to satisfy borrow checker
        let mut updates: Vec<(usize, usize, f32)> = Vec::new();
        
        // For each pair of cells, compute coupling updates
        for i in 0..num_cells {
            let neighbors = self.lattice.cells[i].neighbors;
            let i_flat = i % n;
            
            if i_flat >= pattern.spins.len() { continue; }
            let s_i = pattern.spins[i_flat].value();
            
            // Compute coupling updates to each neighbor
            for (k, &j) in neighbors.iter().enumerate() {
                let j_flat = j % n;
                if j_flat >= pattern.spins.len() { continue; }
                let s_j = pattern.spins[j_flat].value();
                
                // Hebbian update: strengthen connection if spins agree
                // Multiply by 8 to make the pattern couplings strong enough
                let delta = 8.0 * s_i * s_j / n as f32;
                updates.push((i, k, delta));
            }
        }
        
        // Apply all updates
        for (i, k, delta) in updates {
            self.lattice.cells[i].coupling[k] += delta;
        }
        
        self.patterns.push(pattern.clone());
    }
    
    /// Clear all stored patterns
    pub fn clear(&mut self) {
        // Reset all couplings to zero
        for cell in &mut self.lattice.cells {
            cell.coupling = [0.0; 8];
        }
        self.patterns.clear();
    }
    
    /// Set the lattice to a given pattern (as initial state for recall)
    pub fn set_state(&mut self, pattern: &Pattern) {
        let n = self.pattern_size;
        for i in 0..self.lattice.cells.len() {
            let i_flat = i % n;
            if i_flat < pattern.spins.len() {
                self.lattice.cells[i].spin = pattern.spins[i_flat];
            }
        }
    }
    
    /// Get current state as a pattern
    pub fn get_state(&self) -> Pattern {
        let n = self.pattern_size;
        let mut spins = vec![Spin::Down; n];
        
        // Average over z layers
        for i in 0..self.lattice.cells.len() {
            let i_flat = i % n;
            if self.lattice.cells[i].spin == Spin::Up {
                spins[i_flat] = Spin::Up;
            }
        }
        
        Pattern { spins, label: None }
    }
    
    /// Recall: let the lattice settle to nearest stored pattern
    /// 
    /// This is where the magic happens:
    /// - Set initial state to corrupted/partial pattern
    /// - Let the system evolve toward energy minimum
    /// - It naturally falls into the nearest stored pattern
    pub fn recall(&mut self, probe: &Pattern, steps: usize) -> Pattern {
        // Set initial state
        self.set_state(probe);
        
        // Anneal at low temperature (mostly deterministic settling)
        let mut annealer = Annealer::new(0.5, steps);
        annealer.anneal(&mut self.lattice);
        
        self.get_state()
    }
    
    /// Synchronous update: update all spins simultaneously based on local field
    /// More deterministic than annealing, good for clean recall
    pub fn synchronous_update(&mut self, iterations: usize) {
        for _ in 0..iterations {
            // Compute new spins based on current neighbor states
            let mut new_spins = Vec::with_capacity(self.lattice.cells.len());
            
            for i in 0..self.lattice.cells.len() {
                let cell = &self.lattice.cells[i];
                let neighbor_spins = self.lattice.neighbor_spins(i);
                
                // Local field = sum of coupling * neighbor_spin
                let mut field = cell.bias;
                for (k, &ns) in neighbor_spins.iter().enumerate() {
                    field += cell.coupling[k] * ns.value();
                }
                
                // New spin follows sign of local field
                new_spins.push(Spin::from_sign(field));
            }
            
            // Apply all updates simultaneously
            for (i, spin) in new_spins.into_iter().enumerate() {
                self.lattice.cells[i].spin = spin;
            }
        }
    }
    
    /// Find which stored pattern is closest to current state
    pub fn identify(&self) -> Option<(usize, f32, &Pattern)> {
        let current = self.get_state();
        
        self.patterns.iter()
            .enumerate()
            .map(|(i, p)| (i, current.overlap(p), p))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    }
    
    /// Capacity: theoretical number of patterns that can be stored reliably
    /// For Hopfield networks: ~0.14 * N patterns
    pub fn capacity(&self) -> usize {
        (0.14 * self.pattern_size as f32) as usize
    }
}

/// Print a pattern as a 2D grid
pub fn print_pattern(pattern: &Pattern, width: usize) {
    let height = pattern.spins.len() / width;
    
    if let Some(ref label) = pattern.label {
        println!("Pattern: {}", label);
    }
    
    for y in 0..height {
        print!("  ");
        for x in 0..width {
            let idx = y * width + x;
            if idx < pattern.spins.len() {
                match pattern.spins[idx] {
                    Spin::Up => print!("██"),
                    Spin::Down => print!("  "),
                }
            }
        }
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pattern_overlap() {
        let p1 = Pattern::from_bools(&[true, true, false, false], None);
        let p2 = Pattern::from_bools(&[true, true, false, false], None);
        let p3 = Pattern::from_bools(&[false, false, true, true], None);
        
        assert!((p1.overlap(&p2) - 1.0).abs() < 0.01);  // Same pattern
        assert!((p1.overlap(&p3) - (-1.0)).abs() < 0.01);  // Opposite pattern
    }
    
    #[test]
    fn test_pattern_corruption() {
        let p = Pattern::from_bools(&[true; 100], None);
        let corrupted = p.corrupt(0.3);
        
        // Should have some flipped bits
        let overlap = p.overlap(&corrupted);
        assert!(overlap < 1.0);
        assert!(overlap > 0.0);
    }
}
