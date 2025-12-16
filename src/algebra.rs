//! # Concept Algebra: Operations on 3D Thought Forms
//!
//! If concepts are 3D forms, then thinking is form manipulation.
//! 
//! ## Operations
//!
//! - **Addition**: Superimpose forms (combine meanings)
//! - **Subtraction**: Remove one form from another (isolate differences)
//! - **Analogy**: A - B + C = ? (king - man + woman = queen)
//! - **Blend**: Weighted combination with stabilization
//! - **Negate**: Invert the form
//! - **Project**: Extract along an axis (get one semantic dimension)
//!
//! ## The Insight
//!
//! Word2Vec showed: king - man + woman ≈ queen
//! But that's in abstract vector space.
//! 
//! Here, the same operation happens in PHYSICAL configuration space.
//! The forms literally morph and settle.

use crate::concept::{Concept, ConceptMind};
use crate::spin::Spin;

/// Concept Algebra: mathematical operations on 3D thought forms
pub struct ConceptAlgebra {
    /// Size of concept space
    pub size: usize,
}

impl ConceptAlgebra {
    pub fn new(size: usize) -> Self {
        Self { size }
    }
    
    /// Add two concepts: superimpose forms
    /// Where either is Up, result is Up (OR-like)
    pub fn add(&self, a: &Concept, b: &Concept) -> Concept {
        let form: Vec<Spin> = a.form.iter()
            .zip(b.form.iter())
            .map(|(&sa, &sb)| {
                if sa == Spin::Up || sb == Spin::Up {
                    Spin::Up
                } else {
                    Spin::Down
                }
            })
            .collect();
        
        Concept {
            form,
            dims: a.dims,
            label: None,
        }
    }
    
    /// Subtract: remove b's pattern from a
    /// Where a is Up and b is Up, result is Down (removes overlap)
    pub fn subtract(&self, a: &Concept, b: &Concept) -> Concept {
        let form: Vec<Spin> = a.form.iter()
            .zip(b.form.iter())
            .map(|(&sa, &sb)| {
                if sa == Spin::Up && sb == Spin::Down {
                    Spin::Up  // Keep a's unique parts
                } else {
                    Spin::Down
                }
            })
            .collect();
        
        Concept {
            form,
            dims: a.dims,
            label: None,
        }
    }
    
    /// Intersection: where both are Up
    pub fn intersect(&self, a: &Concept, b: &Concept) -> Concept {
        let form: Vec<Spin> = a.form.iter()
            .zip(b.form.iter())
            .map(|(&sa, &sb)| {
                if sa == Spin::Up && sb == Spin::Up {
                    Spin::Up
                } else {
                    Spin::Down
                }
            })
            .collect();
        
        Concept {
            form,
            dims: a.dims,
            label: None,
        }
    }
    
    /// XOR: where exactly one is Up (symmetric difference)
    pub fn xor(&self, a: &Concept, b: &Concept) -> Concept {
        let form: Vec<Spin> = a.form.iter()
            .zip(b.form.iter())
            .map(|(&sa, &sb)| {
                if (sa == Spin::Up) != (sb == Spin::Up) {
                    Spin::Up
                } else {
                    Spin::Down
                }
            })
            .collect();
        
        Concept {
            form,
            dims: a.dims,
            label: None,
        }
    }
    
    /// Negate: flip all spins
    pub fn negate(&self, a: &Concept) -> Concept {
        let form: Vec<Spin> = a.form.iter()
            .map(|&s| s.flipped())
            .collect();
        
        Concept {
            form,
            dims: a.dims,
            label: None,
        }
    }
    
    /// Analogy: A is to B as C is to ?
    /// Computes A - B + C
    pub fn analogy(&self, a: &Concept, b: &Concept, c: &Concept) -> Concept {
        // Remove what B has that A doesn't have
        // Add what C has
        // Result: the "C-equivalent" of A relative to B
        
        // Continuous version for smoother results
        let form: Vec<Spin> = a.form.iter()
            .zip(b.form.iter())
            .zip(c.form.iter())
            .map(|((&sa, &sb), &sc)| {
                let va = sa.value();
                let vb = sb.value();
                let vc = sc.value();
                
                // A - B + C in spin space
                let result = va - vb + vc;
                Spin::from_sign(result)
            })
            .collect();
        
        Concept {
            form,
            dims: a.dims,
            label: None,
        }
    }
    
    /// Analogy with stabilization through thinking
    pub fn analogy_stable(&self, a: &Concept, b: &Concept, c: &Concept, mind: &mut ConceptMind, steps: usize) -> Concept {
        let raw = self.analogy(a, b, c);
        mind.think(&raw, steps)
    }
    
    /// Scale: adjust density by threshold
    pub fn scale(&self, a: &Concept, factor: f32) -> Concept {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let form: Vec<Spin> = a.form.iter()
            .map(|&s| {
                if s == Spin::Up {
                    // Probabilistically keep based on factor
                    if factor >= 1.0 || rng.gen::<f32>() < factor {
                        Spin::Up
                    } else {
                        Spin::Down
                    }
                } else if factor > 1.0 {
                    // Probabilistically add new ups
                    if rng.gen::<f32>() < (factor - 1.0) {
                        Spin::Up
                    } else {
                        Spin::Down
                    }
                } else {
                    Spin::Down
                }
            })
            .collect();
        
        Concept {
            form,
            dims: a.dims,
            label: None,
        }
    }
    
    /// Project onto axis: collapse two dimensions, keep one
    pub fn project(&self, a: &Concept, axis: char) -> Vec<f32> {
        let (dx, dy, dz) = a.dims;
        
        match axis {
            'x' => {
                (0..dx).map(|x| {
                    let mut sum = 0.0f32;
                    for z in 0..dz {
                        for y in 0..dy {
                            sum += a.get(x, y, z).value();
                        }
                    }
                    sum / (dy * dz) as f32
                }).collect()
            }
            'y' => {
                (0..dy).map(|y| {
                    let mut sum = 0.0f32;
                    for z in 0..dz {
                        for x in 0..dx {
                            sum += a.get(x, y, z).value();
                        }
                    }
                    sum / (dx * dz) as f32
                }).collect()
            }
            'z' | _ => {
                (0..dz).map(|z| {
                    let mut sum = 0.0f32;
                    for y in 0..dy {
                        for x in 0..dx {
                            sum += a.get(x, y, z).value();
                        }
                    }
                    sum / (dx * dy) as f32
                }).collect()
            }
        }
    }
    
    /// Rotate concept 90° around axis
    pub fn rotate(&self, a: &Concept, axis: char) -> Concept {
        let (dx, dy, dz) = a.dims;
        let mut result = Concept::empty(dx, dy, dz);
        
        for z in 0..dz {
            for y in 0..dy {
                for x in 0..dx {
                    let spin = a.get(x, y, z);
                    let (nx, ny, nz) = match axis {
                        'x' => (x, dz - 1 - z, y),
                        'y' => (dz - 1 - z, y, x),
                        'z' | _ => (dy - 1 - y, x, z),
                    };
                    if nx < dx && ny < dy && nz < dz {
                        result.set(nx, ny, nz, spin);
                    }
                }
            }
        }
        
        result
    }
    
    /// Translate/shift concept
    pub fn translate(&self, a: &Concept, dx: i32, dy: i32, dz: i32) -> Concept {
        let (sx, sy, sz) = a.dims;
        let mut result = Concept::empty(sx, sy, sz);
        
        for z in 0..sz {
            for y in 0..sy {
                for x in 0..sx {
                    let spin = a.get(x, y, z);
                    if spin == Spin::Up {
                        let nx = (x as i32 + dx).rem_euclid(sx as i32) as usize;
                        let ny = (y as i32 + dy).rem_euclid(sy as i32) as usize;
                        let nz = (z as i32 + dz).rem_euclid(sz as i32) as usize;
                        result.set(nx, ny, nz, Spin::Up);
                    }
                }
            }
        }
        
        result
    }
    
    /// Dilate: expand the form
    pub fn dilate(&self, a: &Concept) -> Concept {
        let (dx, dy, dz) = a.dims;
        let mut result = Concept::empty(dx, dy, dz);
        
        for z in 0..dz {
            for y in 0..dy {
                for x in 0..dx {
                    // If any neighbor is Up, this cell becomes Up
                    let mut has_up_neighbor = a.get(x, y, z) == Spin::Up;
                    
                    for dxi in -1i32..=1 {
                        for dyi in -1i32..=1 {
                            for dzi in -1i32..=1 {
                                let nx = (x as i32 + dxi).rem_euclid(dx as i32) as usize;
                                let ny = (y as i32 + dyi).rem_euclid(dy as i32) as usize;
                                let nz = (z as i32 + dzi).rem_euclid(dz as i32) as usize;
                                if a.get(nx, ny, nz) == Spin::Up {
                                    has_up_neighbor = true;
                                }
                            }
                        }
                    }
                    
                    if has_up_neighbor {
                        result.set(x, y, z, Spin::Up);
                    }
                }
            }
        }
        
        result
    }
    
    /// Erode: shrink the form
    pub fn erode(&self, a: &Concept) -> Concept {
        let (dx, dy, dz) = a.dims;
        let mut result = Concept::empty(dx, dy, dz);
        
        for z in 0..dz {
            for y in 0..dy {
                for x in 0..dx {
                    if a.get(x, y, z) == Spin::Down {
                        continue;
                    }
                    
                    // Only keep if all face-neighbors are also Up
                    let neighbors = [
                        ((x as i32 - 1).rem_euclid(dx as i32) as usize, y, z),
                        ((x + 1) % dx, y, z),
                        (x, (y as i32 - 1).rem_euclid(dy as i32) as usize, z),
                        (x, (y + 1) % dy, z),
                        (x, y, (z as i32 - 1).rem_euclid(dz as i32) as usize),
                        (x, y, (z + 1) % dz),
                    ];
                    
                    let all_up = neighbors.iter()
                        .all(|&(nx, ny, nz)| a.get(nx, ny, nz) == Spin::Up);
                    
                    if all_up {
                        result.set(x, y, z, Spin::Up);
                    }
                }
            }
        }
        
        result
    }
}

/// Demonstrate concept algebra
pub fn demo_concept_algebra() {
    use crate::concept::{ConceptSculptor, visualize_concept};
    
    println!("================================================================");
    println!("           CONCEPT ALGEBRA: Operations on 3D Forms");
    println!("================================================================");
    println!();
    
    let size = 10;
    let algebra = ConceptAlgebra::new(size);
    let sculptor = ConceptSculptor::new(size, size, size);
    
    // Create base concepts
    let sphere = sculptor.sphere(3.5);
    let rod_z = sculptor.rod('z', 2.0);
    let rod_x = sculptor.rod('x', 2.0);
    
    println!("=== BASE CONCEPTS ===\n");
    
    println!("Sphere:");
    visualize_concept(&sphere);
    println!();
    
    println!("Rod (Z-axis):");
    visualize_concept(&rod_z);
    println!();
    
    // Addition
    println!("=== ADDITION: Sphere + Rod ===\n");
    let added = algebra.add(&sphere, &rod_z);
    visualize_concept(&added);
    println!();
    
    // Subtraction
    println!("=== SUBTRACTION: Sphere - Rod ===\n");
    let subtracted = algebra.subtract(&sphere, &rod_z);
    visualize_concept(&subtracted);
    println!();
    
    // Intersection
    println!("=== INTERSECTION: Sphere ∩ Rod ===\n");
    let intersected = algebra.intersect(&sphere, &rod_z);
    visualize_concept(&intersected);
    println!();
    
    // Analogy
    println!("=== ANALOGY: Rod_Z - Sphere + Rod_X = ? ===\n");
    println!("(What is Rod_X's equivalent of Rod_Z relative to Sphere?)\n");
    let analogy_result = algebra.analogy(&rod_z, &sphere, &rod_x);
    visualize_concept(&analogy_result);
    println!();
    
    // Negation
    println!("=== NEGATION: ¬Sphere ===\n");
    let negated = algebra.negate(&sphere);
    visualize_concept(&negated);
    println!();
    
    // Morphological operations
    println!("=== DILATION: Expand Sphere ===\n");
    let dilated = algebra.dilate(&sphere);
    visualize_concept(&dilated);
    println!();
    
    println!("=== EROSION: Shrink Sphere ===\n");
    let eroded = algebra.erode(&sphere);
    visualize_concept(&eroded);
    println!();
    
    // Projection
    println!("=== PROJECTION: Sphere along Z-axis ===\n");
    let projection = algebra.project(&sphere, 'z');
    print!("Z-profile: [");
    for (i, v) in projection.iter().enumerate() {
        if i > 0 { print!(", "); }
        print!("{:.2}", v);
    }
    println!("]");
    println!();
    
    println!("=== THE INSIGHT ===\n");
    println!("Word2Vec: king - man + woman ≈ queen (in abstract vectors)");
    println!("Iron-Net: king - man + woman ≈ queen (in physical 3D forms)");
    println!();
    println!("The analogy operation literally morphs and settles.");
    println!("Thought is geometry. Reasoning is form manipulation.");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::concept::ConceptSculptor;
    
    #[test]
    fn test_add_subtract_inverse() {
        let size = 8;
        let algebra = ConceptAlgebra::new(size);
        let sculptor = ConceptSculptor::new(size, size, size);
        
        let a = sculptor.sphere(3.0);
        let b = sculptor.rod('z', 1.5);
        
        let added = algebra.add(&a, &b);
        let subtracted = algebra.subtract(&added, &b);
        
        // After adding b and subtracting b, should be close to a
        // (not exact due to overlap)
        assert!(a.similarity(&subtracted) > 0.5);
    }
    
    #[test]
    fn test_negation_double() {
        let size = 8;
        let algebra = ConceptAlgebra::new(size);
        let sculptor = ConceptSculptor::new(size, size, size);
        
        let a = sculptor.sphere(3.0);
        let neg_a = algebra.negate(&a);
        let neg_neg_a = algebra.negate(&neg_a);
        
        // Double negation should return original
        assert!((a.similarity(&neg_neg_a) - 1.0).abs() < 0.01);
    }
}
