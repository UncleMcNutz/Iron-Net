//! # Energy: The Currency of Structure
//!
//! Energy is what the lattice minimizes.
//! Lower energy = more stable = more aligned = more "true"
//!
//! In iron:
//! - Aligned spins have lower energy (ferromagnetism)
//! - The system seeks its ground state
//! - Temperature allows exploration of energy landscape
//!
//! In computation:
//! - Energy encodes the problem constraints
//! - Ground state encodes the solution
//! - Annealing finds the answer

use crate::lattice::BCCLattice;

/// Energy functions and calculations for the lattice
pub struct Energy;

impl Energy {
    /// Ising model Hamiltonian for the lattice
    /// H = -Σ J_ij s_i s_j - Σ h_i s_i
    /// 
    /// This IS the problem encoding:
    /// - Set J (couplings) to encode relationships
    /// - Set h (bias) to encode preferences
    /// - Ground state of H is the solution
    pub fn hamiltonian(lattice: &BCCLattice) -> f32 {
        lattice.total_energy()
    }
    
    /// Energy change from flipping spin at given index
    /// This is key for efficient updates - O(1) per spin
    pub fn delta_e(lattice: &BCCLattice, idx: usize) -> f32 {
        let neighbor_spins = lattice.neighbor_spins(idx);
        lattice.cells[idx].flip_energy_delta(&neighbor_spins)
    }
    
    /// Calculate frustration at a cell
    /// Frustration = how much the cell's spin conflicts with neighbors
    /// High frustration = unstable, likely to flip
    pub fn frustration(lattice: &BCCLattice, idx: usize) -> f32 {
        let cell = &lattice.cells[idx];
        let neighbor_spins = lattice.neighbor_spins(idx);
        
        let mut conflict = 0.0;
        for (i, &ns) in neighbor_spins.iter().enumerate() {
            // Conflict when coupling positive but spins opposite
            // or coupling negative but spins same
            let aligned = cell.spin.value() * ns.value();
            let desired = cell.coupling[i].signum();
            if aligned * desired < 0.0 {
                conflict += cell.coupling[i].abs();
            }
        }
        conflict
    }
    
    /// Find the most frustrated cell (most likely to benefit from flip)
    pub fn most_frustrated(lattice: &BCCLattice) -> usize {
        let mut max_frustration = 0.0;
        let mut max_idx = 0;
        
        for i in 0..lattice.cells.len() {
            let f = Self::frustration(lattice, i);
            if f > max_frustration {
                max_frustration = f;
                max_idx = i;
            }
        }
        max_idx
    }
}

/// Temperature schedules for annealing
pub enum CoolingSchedule {
    /// T(t) = T0 * alpha^t
    Exponential { initial: f32, alpha: f32 },
    /// T(t) = T0 / (1 + beta * t)
    Linear { initial: f32, beta: f32 },
    /// T(t) = T0 / log(1 + t)
    Logarithmic { initial: f32 },
}

impl CoolingSchedule {
    /// Get temperature at step t
    pub fn temperature(&self, step: usize) -> f32 {
        let t = step as f32;
        match self {
            CoolingSchedule::Exponential { initial, alpha } => {
                initial * alpha.powf(t)
            }
            CoolingSchedule::Linear { initial, beta } => {
                initial / (1.0 + beta * t)
            }
            CoolingSchedule::Logarithmic { initial } => {
                initial / (1.0 + t).ln()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cooling_schedules() {
        let exp = CoolingSchedule::Exponential { initial: 10.0, alpha: 0.99 };
        assert!(exp.temperature(100) < exp.temperature(0));
        
        let lin = CoolingSchedule::Linear { initial: 10.0, beta: 0.01 };
        assert!(lin.temperature(100) < lin.temperature(0));
    }
}
