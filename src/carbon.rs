//! # Carbon: Interstitial Complexity
//!
//! Pure iron is soft. Add carbon and you get steel.
//! 
//! In the BCC lattice, carbon atoms sit in interstitial sites—
//! the gaps between iron atoms. They don't fit perfectly,
//! which is exactly what makes steel strong.
//!
//! In iron computation:
//! - Regular lattice operations are pure ferrite
//! - Complex operations that don't fit the 8-neighbor pattern
//!   are carbon: necessary but placed carefully
//!
//! Too much carbon makes the steel brittle.
//! Too many special cases make code fragile.
//! Place carbon deliberately.

use crate::spin::Spin;

/// Carbon: operations that don't fit the BCC pattern
/// 
/// Like carbon in steel:
/// - Small amount = strength (useful complexity)
/// - Too much = brittleness (unmaintainable)
/// - Must be carefully placed (interstitial sites only)
#[derive(Clone, Debug)]
pub enum Carbon {
    /// Non-local influence: affect distant cells
    LongRange {
        targets: Vec<usize>,
        strength: f32,
    },
    
    /// Logic gate: compute function of neighbor spins
    Gate {
        function: GateType,
    },
    
    /// Memory: persist state across annealing runs
    Memory {
        stored: Spin,
    },
    
    /// Stochastic: inject controlled randomness
    Noise {
        amplitude: f32,
    },
    
    /// Threshold: activate only when conditions met
    Threshold {
        min_aligned: u8,  // minimum aligned neighbors to activate
    },
}

/// Types of logic gates that can be implemented at carbon sites
#[derive(Clone, Copy, Debug)]
pub enum GateType {
    /// Output = majority of 8 inputs
    Majority,
    /// Output = XOR of all inputs (parity)
    Parity,
    /// Output = 1 iff at least n inputs are 1
    AtLeast(u8),
    /// Output = inverse of majority
    Minority,
}

impl Carbon {
    /// Create a long-range coupling carbon site
    pub fn long_range(targets: Vec<usize>, strength: f32) -> Self {
        Carbon::LongRange { targets, strength }
    }
    
    /// Create a majority-gate carbon site
    pub fn majority_gate() -> Self {
        Carbon::Gate { function: GateType::Majority }
    }
    
    /// Create a parity-gate carbon site  
    pub fn parity_gate() -> Self {
        Carbon::Gate { function: GateType::Parity }
    }
    
    /// Create a memory carbon site
    pub fn memory(initial: Spin) -> Self {
        Carbon::Memory { stored: initial }
    }
    
    /// Create a noise-injection carbon site
    pub fn noise(amplitude: f32) -> Self {
        Carbon::Noise { amplitude }
    }
    
    /// Create a threshold carbon site
    pub fn threshold(min_aligned: u8) -> Self {
        Carbon::Threshold { min_aligned }
    }
    
    /// Compute the output of a gate given neighbor spins
    pub fn compute_gate(gate: GateType, neighbors: &[Spin; 8]) -> Spin {
        match gate {
            GateType::Majority => {
                let sum: i32 = neighbors.iter()
                    .map(|s| if *s == Spin::Up { 1 } else { -1 })
                    .sum();
                if sum >= 0 { Spin::Up } else { Spin::Down }
            }
            GateType::Parity => {
                let count: i32 = neighbors.iter()
                    .filter(|&&s| s == Spin::Up)
                    .count() as i32;
                if count % 2 == 0 { Spin::Down } else { Spin::Up }
            }
            GateType::AtLeast(n) => {
                let count = neighbors.iter()
                    .filter(|&&s| s == Spin::Up)
                    .count();
                if count >= n as usize { Spin::Up } else { Spin::Down }
            }
            GateType::Minority => {
                Carbon::compute_gate(GateType::Majority, neighbors).flipped()
            }
        }
    }
}

/// Steel: a lattice with controlled carbon content
/// 
/// Carbon content determines properties:
/// - 0.0% = pure iron (soft, ductile)
/// - 0.1-0.3% = low carbon steel (structural)
/// - 0.3-0.6% = medium carbon steel (strong)
/// - 0.6-1.0% = high carbon steel (hard)
/// - >1.0% = cast iron (brittle)
pub struct CarbonContent {
    /// Percentage of cells with carbon (0.0 to 1.0)
    pub percentage: f32,
    /// Type of steel this represents
    pub grade: SteelGrade,
}

#[derive(Clone, Copy, Debug)]
pub enum SteelGrade {
    PureIron,      // 0%
    LowCarbon,     // 0.1-0.3%
    MediumCarbon,  // 0.3-0.6%
    HighCarbon,    // 0.6-1.0%
    CastIron,      // >1%
}

impl CarbonContent {
    pub fn classify(percentage: f32) -> SteelGrade {
        match percentage {
            p if p < 0.001 => SteelGrade::PureIron,
            p if p < 0.003 => SteelGrade::LowCarbon,
            p if p < 0.006 => SteelGrade::MediumCarbon,
            p if p < 0.010 => SteelGrade::HighCarbon,
            _ => SteelGrade::CastIron,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_majority_gate() {
        let up_majority = [Spin::Up, Spin::Up, Spin::Up, Spin::Up, 
                          Spin::Up, Spin::Down, Spin::Down, Spin::Down];
        assert_eq!(Carbon::compute_gate(GateType::Majority, &up_majority), Spin::Up);
        
        let down_majority = [Spin::Down, Spin::Down, Spin::Down, Spin::Down,
                            Spin::Down, Spin::Up, Spin::Up, Spin::Up];
        assert_eq!(Carbon::compute_gate(GateType::Majority, &down_majority), Spin::Down);
    }
    
    #[test]
    fn test_parity_gate() {
        let even = [Spin::Up, Spin::Up, Spin::Up, Spin::Up,
                   Spin::Down, Spin::Down, Spin::Down, Spin::Down];
        assert_eq!(Carbon::compute_gate(GateType::Parity, &even), Spin::Down);
        
        let odd = [Spin::Up, Spin::Up, Spin::Up, Spin::Up,
                  Spin::Up, Spin::Down, Spin::Down, Spin::Down];
        assert_eq!(Carbon::compute_gate(GateType::Parity, &odd), Spin::Up);
    }
    
    #[test]
    fn test_steel_grades() {
        assert!(matches!(CarbonContent::classify(0.0), SteelGrade::PureIron));
        assert!(matches!(CarbonContent::classify(0.002), SteelGrade::LowCarbon));
        assert!(matches!(CarbonContent::classify(0.005), SteelGrade::MediumCarbon));
        assert!(matches!(CarbonContent::classify(0.008), SteelGrade::HighCarbon));
        assert!(matches!(CarbonContent::classify(0.02), SteelGrade::CastIron));
    }
}
