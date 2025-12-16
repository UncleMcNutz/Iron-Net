//! # Copper: FCC Lattice for Transmission
//!
//! If Iron (BCC) is for holding and thinking,
//! Copper (FCC) is for flowing and transmitting.
//!
//! ## Crystal Structure
//!
//! Face-Centered Cubic (FCC):
//! - Atoms at cube corners AND face centers
//! - Each atom has 12 nearest neighbors (vs Iron's 8)
//! - Higher coordination = better flow paths
//! - More ductile = easier to reshape routes
//!
//! ```text
//!         ●───────────●
//!        /|    ◆     /|      ● = Corner atoms
//!       / |   /     / |      ◆ = Face-center atoms
//!      ●───────────●  |
//!      |◆ |       |◆ |      12 nearest neighbors:
//!      |  ●───◆───|──●      - 4 in same plane
//!      | /   \    | /       - 4 above
//!      |/    ◆    |/        - 4 below
//!      ●───────────●
//! ```
//!
//! ## Copper vs Iron
//!
//! Iron (BCC):
//!   - Magnetic: holds spin states
//!   - 8 neighbors: configuration space
//!   - Ferromagnetic: aligns with neighbors
//!   - PURPOSE: Store, settle, remember
//!
//! Copper (FCC):
//!   - Non-magnetic: no persistent state
//!   - 12 neighbors: more flow paths
//!   - Diamagnetic: repels, keeps moving
//!   - PURPOSE: Route, transmit, connect

use rand::Rng;

/// A node in the FCC lattice - optimized for transmission
#[derive(Clone, Debug)]
pub struct CopperNode {
    /// Current signal value (not spin - continuous)
    pub signal: f32,
    /// Indices of 12 nearest neighbors
    pub neighbors: [usize; 12],
    /// Conductance to each neighbor (how easily signal flows)
    pub conductance: [f32; 12],
    /// Is this node currently active (carrying signal)?
    pub active: bool,
}

impl CopperNode {
    pub fn new() -> Self {
        Self {
            signal: 0.0,
            neighbors: [0; 12],
            conductance: [1.0; 12],  // Full conductance by default
            active: false,
        }
    }
}

/// FCC Lattice - the copper transmission substrate
pub struct FCCLattice {
    /// All nodes in the lattice
    pub nodes: Vec<CopperNode>,
    /// Dimensions
    pub size_x: usize,
    pub size_y: usize,
    pub size_z: usize,
}

impl FCCLattice {
    /// Create a new FCC lattice
    pub fn new(size_x: usize, size_y: usize, size_z: usize) -> Self {
        // FCC has 4 atoms per unit cell, but we'll use a simplified model
        // where each grid point is a node with 12-neighbor connectivity
        let total = size_x * size_y * size_z;
        let mut nodes = vec![CopperNode::new(); total];
        
        // Compute 12-neighbor connectivity for FCC
        for z in 0..size_z {
            for y in 0..size_y {
                for x in 0..size_x {
                    let idx = x + y * size_x + z * size_x * size_y;
                    nodes[idx].neighbors = Self::compute_fcc_neighbors(
                        x, y, z, size_x, size_y, size_z
                    );
                }
            }
        }
        
        Self { nodes, size_x, size_y, size_z }
    }
    
    /// Compute the 12 nearest neighbors in FCC structure
    /// FCC neighbors are at positions like (±1,±1,0), (±1,0,±1), (0,±1,±1)
    fn compute_fcc_neighbors(
        x: usize, y: usize, z: usize,
        sx: usize, sy: usize, sz: usize
    ) -> [usize; 12] {
        let wrap = |v: i32, max: usize| -> usize {
            ((v % max as i32) + max as i32) as usize % max
        };
        
        let idx = |dx: i32, dy: i32, dz: i32| -> usize {
            let nx = wrap(x as i32 + dx, sx);
            let ny = wrap(y as i32 + dy, sy);
            let nz = wrap(z as i32 + dz, sz);
            nx + ny * sx + nz * sx * sy
        };
        
        // 12 FCC nearest neighbors
        [
            // 4 neighbors in xy plane (z=0)
            idx(1, 1, 0),
            idx(1, -1, 0),
            idx(-1, 1, 0),
            idx(-1, -1, 0),
            // 4 neighbors in xz plane (y=0)
            idx(1, 0, 1),
            idx(1, 0, -1),
            idx(-1, 0, 1),
            idx(-1, 0, -1),
            // 4 neighbors in yz plane (x=0)
            idx(0, 1, 1),
            idx(0, 1, -1),
            idx(0, -1, 1),
            idx(0, -1, -1),
        ]
    }
    
    /// Get node at (x, y, z)
    pub fn get(&self, x: usize, y: usize, z: usize) -> &CopperNode {
        let idx = x + y * self.size_x + z * self.size_x * self.size_y;
        &self.nodes[idx]
    }
    
    /// Get mutable node at (x, y, z)
    pub fn get_mut(&mut self, x: usize, y: usize, z: usize) -> &mut CopperNode {
        let idx = x + y * self.size_x + z * self.size_x * self.size_y;
        &mut self.nodes[idx]
    }
    
    /// Get coordinates from index
    pub fn coords(&self, idx: usize) -> (usize, usize, usize) {
        let z = idx / (self.size_x * self.size_y);
        let rem = idx % (self.size_x * self.size_y);
        let y = rem / self.size_x;
        let x = rem % self.size_x;
        (x, y, z)
    }
    
    /// Total nodes
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    
    /// Inject signal at a position
    pub fn inject(&mut self, x: usize, y: usize, z: usize, signal: f32) {
        let node = self.get_mut(x, y, z);
        node.signal = signal;
        node.active = signal.abs() > 0.01;
    }
    
    /// Propagate signals one step (diffusion-like)
    /// Unlike Iron which settles, Copper flows
    pub fn propagate(&mut self, damping: f32) {
        let n = self.nodes.len();
        let mut new_signals = vec![0.0f32; n];
        
        for i in 0..n {
            let node = &self.nodes[i];
            let mut incoming = 0.0f32;
            let mut total_conductance = 0.0f32;
            
            // Sum signals from neighbors weighted by conductance
            for k in 0..12 {
                let j = node.neighbors[k];
                let c = node.conductance[k];
                incoming += self.nodes[j].signal * c;
                total_conductance += c;
            }
            
            // Average incoming signal (diffusion)
            if total_conductance > 0.0 {
                new_signals[i] = (incoming / total_conductance) * damping;
            }
        }
        
        // Update all signals
        for (i, &new_sig) in new_signals.iter().enumerate() {
            self.nodes[i].signal = new_sig;
            self.nodes[i].active = new_sig.abs() > 0.01;
        }
    }
    
    /// Count active nodes
    pub fn active_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.active).count()
    }
    
    /// Total signal in system
    pub fn total_signal(&self) -> f32 {
        self.nodes.iter().map(|n| n.signal).sum()
    }
    
    /// Get signals as vector
    pub fn signals(&self) -> Vec<f32> {
        self.nodes.iter().map(|n| n.signal).collect()
    }
    
    /// Set signals from vector
    pub fn set_signals(&mut self, signals: &[f32]) {
        for (i, &s) in signals.iter().enumerate() {
            if i < self.nodes.len() {
                self.nodes[i].signal = s;
                self.nodes[i].active = s.abs() > 0.01;
            }
        }
    }
    
    /// Clear all signals
    pub fn clear(&mut self) {
        for node in &mut self.nodes {
            node.signal = 0.0;
            node.active = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fcc_12_neighbors() {
        let lattice = FCCLattice::new(4, 4, 4);
        // Each node should have 12 unique neighbors
        let node = &lattice.nodes[0];
        let mut unique: Vec<usize> = node.neighbors.to_vec();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), 12);
    }
    
    #[test]
    fn test_signal_propagation() {
        let mut lattice = FCCLattice::new(5, 5, 5);
        
        // Inject signal at center
        lattice.inject(2, 2, 2, 1.0);
        assert_eq!(lattice.active_count(), 1);
        
        // Propagate
        lattice.propagate(0.9);
        
        // Signal should have spread
        assert!(lattice.active_count() > 1);
    }
}
