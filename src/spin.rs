//! # Spin: The Fundamental State
//!
//! In iron, each atom has a magnetic moment - a spin.
//! Spins can point up (+1) or down (-1).
//! 
//! This is the atomic unit of computation:
//! - Not a bit imposed on silicon
//! - A magnetic state natural to iron
//! 
//! The spin wants to align with its neighbors.
//! From this simple desire, complex patterns emerge.

use rand::Rng;

/// Magnetic spin state: the atomic bit of iron computation
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spin {
    Up,
    Down,
}

impl Spin {
    /// Random spin with equal probability
    pub fn random() -> Self {
        if rand::thread_rng().gen_bool(0.5) {
            Spin::Up
        } else {
            Spin::Down
        }
    }
    
    /// Numerical value: +1 for up, -1 for down
    #[inline]
    pub fn value(&self) -> f32 {
        match self {
            Spin::Up => 1.0,
            Spin::Down => -1.0,
        }
    }
    
    /// Flip the spin
    #[inline]
    pub fn flip(&mut self) {
        *self = match self {
            Spin::Up => Spin::Down,
            Spin::Down => Spin::Up,
        }
    }
    
    /// Return the flipped spin without modifying
    #[inline]
    pub fn flipped(&self) -> Self {
        match self {
            Spin::Up => Spin::Down,
            Spin::Down => Spin::Up,
        }
    }
    
    /// Create spin from sign: positive = Up, negative/zero = Down
    pub fn from_sign(x: f32) -> Self {
        if x > 0.0 {
            Spin::Up
        } else {
            Spin::Down
        }
    }
    
    /// Create spin from boolean: true = Up, false = Down
    pub fn from_bool(b: bool) -> Self {
        if b { Spin::Up } else { Spin::Down }
    }
    
    /// Convert to boolean: Up = true, Down = false
    pub fn to_bool(&self) -> bool {
        matches!(self, Spin::Up)
    }
}

impl std::fmt::Display for Spin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Spin::Up => write!(f, "↑"),
            Spin::Down => write!(f, "↓"),
        }
    }
}

impl From<bool> for Spin {
    fn from(b: bool) -> Self {
        Spin::from_bool(b)
    }
}

impl From<Spin> for bool {
    fn from(s: Spin) -> Self {
        s.to_bool()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_spin_values() {
        assert_eq!(Spin::Up.value(), 1.0);
        assert_eq!(Spin::Down.value(), -1.0);
    }
    
    #[test]
    fn test_spin_flip() {
        let mut s = Spin::Up;
        s.flip();
        assert_eq!(s, Spin::Down);
        s.flip();
        assert_eq!(s, Spin::Up);
    }
    
    #[test]
    fn test_spin_bool_conversion() {
        assert_eq!(Spin::from_bool(true), Spin::Up);
        assert_eq!(Spin::from_bool(false), Spin::Down);
        assert!(Spin::Up.to_bool());
        assert!(!Spin::Down.to_bool());
    }
}
