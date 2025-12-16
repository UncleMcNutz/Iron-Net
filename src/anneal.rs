//! # Annealing: Learning Through Cooling
//!
//! Simulated annealing mirrors the metallurgical process:
//! - Heat metal to allow atomic rearrangement
//! - Slowly cool to let it find stable structure
//! - Fast cooling = amorphous/glassy (stuck in local minima)
//! - Slow cooling = crystalline (global optimum)
//!
//! For iron computation:
//! - High temperature: explore random configurations
//! - Lower temperature: settle into meaningful patterns
//! - Learning = annealing with constrained inputs
//!
//! The Metropolis algorithm is our hammer and anvil.

use crate::lattice::BCCLattice;
use crate::energy::{Energy, CoolingSchedule};
use rand::Rng;

/// Annealing engine for the BCC lattice
pub struct Annealer {
    /// Cooling schedule
    pub schedule: CoolingSchedule,
    /// Current step in the annealing process
    pub step: usize,
    /// Maximum steps
    pub max_steps: usize,
}

impl Annealer {
    /// Create a new annealer with exponential cooling
    pub fn new(initial_temp: f32, max_steps: usize) -> Self {
        // Alpha chosen so temperature is ~0.01 * initial at end
        let alpha = (0.01_f32).powf(1.0 / max_steps as f32);
        Self {
            schedule: CoolingSchedule::Exponential { initial: initial_temp, alpha },
            step: 0,
            max_steps,
        }
    }
    
    /// Current temperature
    pub fn temperature(&self) -> f32 {
        self.schedule.temperature(self.step)
    }
    
    /// Perform one Monte Carlo step
    /// Returns true if a spin was flipped
    pub fn step(&mut self, lattice: &mut BCCLattice) -> bool {
        let mut rng = rand::thread_rng();
        let temp = self.temperature();
        
        // Pick random cell
        let idx = rng.gen_range(0..lattice.cells.len());
        
        // Skip if cell is strongly pinned (input boundary)
        if lattice.cells[idx].bias.abs() > 10.0 {
            self.step += 1;
            return false;
        }
        
        // Calculate energy change
        let delta_e = Energy::delta_e(lattice, idx);
        
        // Metropolis criterion:
        // Accept if energy decreases, or with probability exp(-ΔE/T) if increases
        let accept = if delta_e <= 0.0 {
            true
        } else if temp > 0.0 {
            rng.gen::<f32>() < (-delta_e / temp).exp()
        } else {
            false
        };
        
        if accept {
            lattice.cells[idx].spin.flip();
        }
        
        self.step += 1;
        accept
    }
    
    /// Run a full sweep (one attempted flip per cell on average)
    pub fn sweep(&mut self, lattice: &mut BCCLattice) -> usize {
        let n = lattice.cells.len();
        let mut flips = 0;
        for _ in 0..n {
            if self.step(lattice) {
                flips += 1;
            }
        }
        flips
    }
    
    /// Run full annealing process
    /// Returns (final_energy, final_magnetization)
    pub fn anneal(&mut self, lattice: &mut BCCLattice) -> (f32, f32) {
        while self.step < self.max_steps {
            self.sweep(lattice);
        }
        (Energy::hamiltonian(lattice), lattice.magnetization())
    }
    
    /// Reset annealer for new run
    pub fn reset(&mut self) {
        self.step = 0;
    }
    
    /// Is annealing complete?
    pub fn is_done(&self) -> bool {
        self.step >= self.max_steps
    }
}

/// Parallel annealing using checkerboard decomposition
/// Red and black cells can update simultaneously (no neighbor conflicts)
pub fn parallel_sweep(lattice: &mut BCCLattice, temperature: f32) -> usize {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut flips = 0;
    
    // In BCC, we can update cells where (x+y+z) % 2 == 0 independently,
    // then update cells where (x+y+z) % 2 == 1
    
    for parity in 0..2 {
        for z in 0..lattice.size_z {
            for y in 0..lattice.size_y {
                for x in 0..lattice.size_x {
                    if (x + y + z) % 2 != parity {
                        continue;
                    }
                    
                    let idx = lattice.index(x, y, z);
                    
                    // Skip pinned cells
                    if lattice.cells[idx].bias.abs() > 10.0 {
                        continue;
                    }
                    
                    let delta_e = Energy::delta_e(lattice, idx);
                    
                    let accept = if delta_e <= 0.0 {
                        true
                    } else if temperature > 0.0 {
                        rng.gen::<f32>() < (-delta_e / temperature).exp()
                    } else {
                        false
                    };
                    
                    if accept {
                        lattice.cells[idx].spin.flip();
                        flips += 1;
                    }
                }
            }
        }
    }
    
    flips
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_annealing_reduces_energy() {
        let mut lattice = BCCLattice::new(8, 8, 8);
        let initial_energy = Energy::hamiltonian(&lattice);
        
        let mut annealer = Annealer::new(10.0, 10000);
        let (final_energy, _) = annealer.anneal(&mut lattice);
        
        // Energy should decrease (become more negative)
        assert!(final_energy <= initial_energy);
    }
    
    #[test]
    fn test_low_temp_produces_domains() {
        let mut lattice = BCCLattice::new(8, 8, 8);
        let mut annealer = Annealer::new(10.0, 50000);
        annealer.anneal(&mut lattice);
        
        // At low temperature, magnetization should be non-zero
        // (system forms domains, not random)
        let mag = lattice.magnetization().abs();
        assert!(mag > 0.5, "Expected domain formation, got magnetization {}", mag);
    }
}
