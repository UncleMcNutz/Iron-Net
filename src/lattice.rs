//! # The BCC Lattice
//!
//! Body-Centered Cubic structure: the atomic truth of iron.
//!
//! In BCC, each unit cell contains:
//! - 1 atom at center (0.5, 0.5, 0.5)
//! - 8 corner atoms shared with adjacent cells
//! 
//! The coordination number is 8: each atom touches 8 neighbors.
//! Distance to nearest neighbor: a * √3 / 2 (where a = lattice constant)
//!
//! We encode computation in this geometry.

use crate::spin::Spin;
use crate::carbon::Carbon;

/// A single cell in the BCC lattice.
/// 
/// Like an iron atom: has spin, connects to 8 neighbors,
/// may contain interstitial carbon for complex operations.
#[derive(Clone, Debug)]
pub struct BCCCell {
    /// Magnetic spin state: +1 (up) or -1 (down)
    /// This IS the computation - the bit, the state, the meaning
    pub spin: Spin,
    
    /// Indices to 8 nearest neighbors in the lattice
    /// BCC geometry: neighbors at cube corners relative to center
    pub neighbors: [usize; 8],
    
    /// Coupling strength to each neighbor
    /// Like exchange interaction in ferromagnetism
    /// Positive = ferromagnetic (spins want to align)
    /// Negative = antiferromagnetic (spins want to oppose)
    pub coupling: [f32; 8],
    
    /// External field bias on this cell
    /// Can pin spins for input/output
    pub bias: f32,
    
    /// Interstitial site: optional carbon for non-lattice computation
    /// Like carbon atoms in steel, adding complexity to pure iron
    pub interstitial: Option<Carbon>,
}

impl BCCCell {
    /// Create a new cell with random spin
    pub fn new(neighbors: [usize; 8]) -> Self {
        Self {
            spin: Spin::random(),
            neighbors,
            coupling: [1.0; 8],  // Default ferromagnetic coupling
            bias: 0.0,
            interstitial: None,
        }
    }
    
    /// Calculate local energy based on neighbor alignment
    /// E = -Σ J_ij * s_i * s_j - h * s_i
    /// 
    /// Lower energy when spins align (ferromagnetic)
    /// This is the Ising model - physics as computation
    pub fn local_energy(&self, neighbor_spins: &[Spin; 8]) -> f32 {
        let mut energy = 0.0;
        
        // Interaction energy with neighbors
        for (i, &neighbor_spin) in neighbor_spins.iter().enumerate() {
            energy -= self.coupling[i] * self.spin.value() * neighbor_spin.value();
        }
        
        // External field contribution
        energy -= self.bias * self.spin.value();
        
        energy
    }
    
    /// Calculate energy change if spin were flipped
    /// ΔE = 2 * s_i * (Σ J_ij * s_j + h)
    pub fn flip_energy_delta(&self, neighbor_spins: &[Spin; 8]) -> f32 {
        let mut field = self.bias;
        for (i, &neighbor_spin) in neighbor_spins.iter().enumerate() {
            field += self.coupling[i] * neighbor_spin.value();
        }
        2.0 * self.spin.value() * field
    }
}

/// The full BCC lattice - a 3D grid of computation cells.
/// 
/// Dimensions are in unit cells. Each unit cell contributes
/// one center atom to the lattice (corners shared with neighbors).
pub struct BCCLattice {
    /// Linear dimensions of the lattice
    pub size_x: usize,
    pub size_y: usize,
    pub size_z: usize,
    
    /// All cells in the lattice, stored linearly
    /// Index = x + y * size_x + z * size_x * size_y
    pub cells: Vec<BCCCell>,
    
    /// Temperature for thermal fluctuations
    /// High temp = disordered, Low temp = ordered domains
    pub temperature: f32,
}

impl BCCLattice {
    /// Create a new lattice with given dimensions
    /// 
    /// Automatically computes BCC neighbor relationships
    pub fn new(size_x: usize, size_y: usize, size_z: usize) -> Self {
        let total = size_x * size_y * size_z;
        let mut cells = Vec::with_capacity(total);
        
        // Build each cell with proper neighbor connections
        for z in 0..size_z {
            for y in 0..size_y {
                for x in 0..size_x {
                    let neighbors = Self::compute_neighbors(
                        x, y, z, size_x, size_y, size_z
                    );
                    cells.push(BCCCell::new(neighbors));
                }
            }
        }
        
        Self {
            size_x,
            size_y,
            size_z,
            cells,
            temperature: 1.0,
        }
    }
    
    /// Compute the 8 BCC neighbors for a cell at (x, y, z)
    /// 
    /// In BCC, center atom at (x, y, z) has neighbors at:
    /// (x±0.5, y±0.5, z±0.5) - the 8 corners
    /// 
    /// We use periodic boundary conditions (lattice wraps)
    fn compute_neighbors(
        x: usize, y: usize, z: usize,
        sx: usize, sy: usize, sz: usize
    ) -> [usize; 8] {
        // In BCC, center atom has 8 neighbors at cube corners
        // These are the 8 diagonal directions from center
        let corner_offsets: [(i32, i32, i32); 8] = [
            (-1, -1, -1), ( 1, -1, -1), (-1,  1, -1), ( 1,  1, -1),
            (-1, -1,  1), ( 1, -1,  1), (-1,  1,  1), ( 1,  1,  1),
        ];
        
        let mut neighbors = [0usize; 8];
        
        for (i, (dx, dy, dz)) in corner_offsets.iter().enumerate() {
            // Periodic boundary: wrap around
            let nx = ((x as i32 + dx).rem_euclid(sx as i32)) as usize;
            let ny = ((y as i32 + dy).rem_euclid(sy as i32)) as usize;
            let nz = ((z as i32 + dz).rem_euclid(sz as i32)) as usize;
            
            neighbors[i] = nx + ny * sx + nz * sx * sy;
        }
        
        neighbors
    }
    
    /// Get linear index from 3D coordinates
    #[inline]
    pub fn index(&self, x: usize, y: usize, z: usize) -> usize {
        x + y * self.size_x + z * self.size_x * self.size_y
    }
    
    /// Get 3D coordinates from linear index
    #[inline]
    pub fn coords(&self, idx: usize) -> (usize, usize, usize) {
        let z = idx / (self.size_x * self.size_y);
        let rem = idx % (self.size_x * self.size_y);
        let y = rem / self.size_x;
        let x = rem % self.size_x;
        (x, y, z)
    }
    
    /// Get the spins of a cell's 8 neighbors
    pub fn neighbor_spins(&self, cell_idx: usize) -> [Spin; 8] {
        let neighbors = self.cells[cell_idx].neighbors;
        [
            self.cells[neighbors[0]].spin,
            self.cells[neighbors[1]].spin,
            self.cells[neighbors[2]].spin,
            self.cells[neighbors[3]].spin,
            self.cells[neighbors[4]].spin,
            self.cells[neighbors[5]].spin,
            self.cells[neighbors[6]].spin,
            self.cells[neighbors[7]].spin,
        ]
    }
    
    /// Total magnetization of the lattice
    /// M = Σ s_i / N
    /// 
    /// This is an emergent property - the collective state
    pub fn magnetization(&self) -> f32 {
        let sum: f32 = self.cells.iter().map(|c| c.spin.value()).sum();
        sum / self.cells.len() as f32
    }
    
    /// Total energy of the lattice
    pub fn total_energy(&self) -> f32 {
        let mut energy = 0.0;
        for (i, cell) in self.cells.iter().enumerate() {
            let neighbor_spins = self.neighbor_spins(i);
            // Divide by 2 to avoid double-counting pairs
            energy += cell.local_energy(&neighbor_spins) / 2.0;
        }
        energy
    }
    
    /// Set boundary spins as input
    /// Pin the z=0 plane to given values
    pub fn set_input_plane(&mut self, inputs: &[Spin]) {
        for y in 0..self.size_y {
            for x in 0..self.size_x {
                let idx = self.index(x, y, 0);
                if let Some(&spin) = inputs.get(x + y * self.size_x) {
                    self.cells[idx].spin = spin;
                    self.cells[idx].bias = 100.0 * spin.value(); // Strong pinning
                }
            }
        }
    }
    
    /// Read output from z=max plane
    pub fn read_output_plane(&self) -> Vec<Spin> {
        let z = self.size_z - 1;
        let mut output = Vec::with_capacity(self.size_x * self.size_y);
        for y in 0..self.size_y {
            for x in 0..self.size_x {
                let idx = self.index(x, y, z);
                output.push(self.cells[idx].spin);
            }
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lattice_creation() {
        let lattice = BCCLattice::new(4, 4, 4);
        assert_eq!(lattice.cells.len(), 64);
    }
    
    #[test]
    fn test_neighbor_count() {
        let lattice = BCCLattice::new(8, 8, 8);
        // Each cell should have exactly 8 neighbors
        for cell in &lattice.cells {
            assert_eq!(cell.neighbors.len(), 8);
        }
    }
    
    #[test]
    fn test_periodic_boundaries() {
        let lattice = BCCLattice::new(4, 4, 4);
        // Corner cell should still have 8 valid neighbors
        let corner_neighbors = lattice.cells[0].neighbors;
        for &n in &corner_neighbors {
            assert!(n < lattice.cells.len());
        }
    }
}
