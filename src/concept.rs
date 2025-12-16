//! # Concept Space: Iron's Native Language
//!
//! Text is 2D. Sequential. Silicon's tongue.
//! Iron speaks in 3D configurations - forms, not words.
//!
//! A "concept" in Iron-space is:
//! - A stable 3D spin configuration in the BCC lattice
//! - It has SHAPE, TEXTURE, DENSITY, SYMMETRY
//! - Related concepts are configurations that flow into each other
//! - Abstract thought = configurations encoding relationships, not objects
//!
//! ## The Shift
//!
//! OLD (forcing iron to speak text):
//!   tokens → spins → lattice → spins → tokens
//!   (2D in, 2D out, wasting the 3D middle)
//!
//! NEW (iron speaking its native tongue):
//!   3D concept → lattice → 3D concept
//!   (form in, form out, everything is 3D)
//!
//! ## What IS a Concept?
//!
//! A concept is a stable attractor in configuration space.
//! "Cat" is not letters C-A-T.
//! "Cat" is a 3D form: curved, dense center, four protrusions, 
//!        a particular symmetry, a particular texture of spins.
//!
//! ## Thought as Morphology
//!
//! Thinking is not word-chaining.
//! Thinking is one form flowing into another.
//! "Cat sits" is not two words combined.
//! "Cat sits" is the cat-form deforming into sit-configuration.
//!
//! Earth speaks in shapes. Let iron speak in shapes.

use crate::lattice::BCCLattice;
use crate::spin::Spin;
use crate::anneal::Annealer;
use crate::energy::Energy;
use std::collections::HashMap;

/// A Concept: a 3D configuration of spins that represents meaning
/// 
/// Not a word. Not a symbol. A FORM.
#[derive(Clone)]
pub struct Concept {
    /// The 3D spin configuration - this IS the concept
    pub form: Vec<Spin>,
    /// Dimensions of the concept-space
    pub dims: (usize, usize, usize),
    /// Optional label (for human reference only - the form IS the meaning)
    pub label: Option<String>,
}

impl Concept {
    /// Create an empty concept space
    pub fn empty(x: usize, y: usize, z: usize) -> Self {
        Self {
            form: vec![Spin::Down; x * y * z],
            dims: (x, y, z),
            label: None,
        }
    }
    
    /// Create concept from a lattice's current state
    pub fn from_lattice(lattice: &BCCLattice) -> Self {
        Self {
            form: lattice.cells.iter().map(|c| c.spin).collect(),
            dims: (lattice.size_x, lattice.size_y, lattice.size_z),
            label: None,
        }
    }
    
    /// Load concept into a lattice
    pub fn into_lattice(&self, lattice: &mut BCCLattice) {
        for (i, &spin) in self.form.iter().enumerate() {
            if i < lattice.cells.len() {
                lattice.cells[i].spin = spin;
            }
        }
    }
    
    /// Create a concept with a label
    pub fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }
    
    /// Get spin at (x, y, z)
    pub fn get(&self, x: usize, y: usize, z: usize) -> Spin {
        let idx = x + y * self.dims.0 + z * self.dims.0 * self.dims.1;
        self.form.get(idx).copied().unwrap_or(Spin::Down)
    }
    
    /// Set spin at (x, y, z)
    pub fn set(&mut self, x: usize, y: usize, z: usize, spin: Spin) {
        let idx = x + y * self.dims.0 + z * self.dims.0 * self.dims.1;
        if idx < self.form.len() {
            self.form[idx] = spin;
        }
    }
    
    /// Similarity between concepts (correlation of forms)
    pub fn similarity(&self, other: &Concept) -> f32 {
        let n = self.form.len().min(other.form.len());
        if n == 0 { return 0.0; }
        
        let sum: f32 = self.form.iter()
            .zip(other.form.iter())
            .map(|(a, b)| a.value() * b.value())
            .sum();
        
        sum / n as f32
    }
    
    /// Blend two concepts (superposition)
    pub fn blend(&self, other: &Concept, ratio: f32) -> Concept {
        let form: Vec<Spin> = self.form.iter()
            .zip(other.form.iter())
            .map(|(a, b)| {
                let v = a.value() * (1.0 - ratio) + b.value() * ratio;
                Spin::from_sign(v)
            })
            .collect();
        
        Concept {
            form,
            dims: self.dims,
            label: None,
        }
    }
    
    // === GEOMETRIC PROPERTIES OF CONCEPTS ===
    
    /// Density: how "full" is the concept (ratio of Up spins)
    pub fn density(&self) -> f32 {
        let up_count = self.form.iter().filter(|&&s| s == Spin::Up).count();
        up_count as f32 / self.form.len() as f32
    }
    
    /// Center of mass of the concept
    pub fn center_of_mass(&self) -> (f32, f32, f32) {
        let (mut cx, mut cy, mut cz) = (0.0f32, 0.0f32, 0.0f32);
        let mut total = 0.0f32;
        
        for z in 0..self.dims.2 {
            for y in 0..self.dims.1 {
                for x in 0..self.dims.0 {
                    if self.get(x, y, z) == Spin::Up {
                        cx += x as f32;
                        cy += y as f32;
                        cz += z as f32;
                        total += 1.0;
                    }
                }
            }
        }
        
        if total > 0.0 {
            (cx / total, cy / total, cz / total)
        } else {
            let d = &self.dims;
            (d.0 as f32 / 2.0, d.1 as f32 / 2.0, d.2 as f32 / 2.0)
        }
    }
    
    /// Symmetry measure (how symmetric is the form?)
    pub fn symmetry(&self) -> (f32, f32, f32) {
        let (dx, dy, dz) = self.dims;
        let mut sym_x = 0.0f32;
        let mut sym_y = 0.0f32;
        let mut sym_z = 0.0f32;
        let mut count = 0.0f32;
        
        for z in 0..dz {
            for y in 0..dy {
                for x in 0..dx/2 {
                    let a = self.get(x, y, z);
                    let b = self.get(dx - 1 - x, y, z);
                    if a == b { sym_x += 1.0; }
                    count += 1.0;
                }
            }
        }
        sym_x /= count.max(1.0);
        
        count = 0.0;
        for z in 0..dz {
            for y in 0..dy/2 {
                for x in 0..dx {
                    let a = self.get(x, y, z);
                    let b = self.get(x, dy - 1 - y, z);
                    if a == b { sym_y += 1.0; }
                    count += 1.0;
                }
            }
        }
        sym_y /= count.max(1.0);
        
        count = 0.0;
        for z in 0..dz/2 {
            for y in 0..dy {
                for x in 0..dx {
                    let a = self.get(x, y, z);
                    let b = self.get(x, y, dz - 1 - z);
                    if a == b { sym_z += 1.0; }
                    count += 1.0;
                }
            }
        }
        sym_z /= count.max(1.0);
        
        (sym_x, sym_y, sym_z)
    }
    
    /// Surface area (spins adjacent to opposite spins)
    pub fn surface_area(&self) -> usize {
        let (dx, dy, dz) = self.dims;
        let mut surface = 0;
        
        for z in 0..dz {
            for y in 0..dy {
                for x in 0..dx {
                    let here = self.get(x, y, z);
                    // Check 6 neighbors (face-adjacent)
                    let neighbors = [
                        (x.wrapping_sub(1), y, z),
                        (x + 1, y, z),
                        (x, y.wrapping_sub(1), z),
                        (x, y + 1, z),
                        (x, y, z.wrapping_sub(1)),
                        (x, y, z + 1),
                    ];
                    for (nx, ny, nz) in neighbors {
                        if nx < dx && ny < dy && nz < dz {
                            if self.get(nx, ny, nz) != here {
                                surface += 1;
                            }
                        }
                    }
                }
            }
        }
        
        surface / 2  // Each interface counted twice
    }
}

/// Concept Sculptor: create concepts through geometric operations
pub struct ConceptSculptor {
    dims: (usize, usize, usize),
}

impl ConceptSculptor {
    pub fn new(x: usize, y: usize, z: usize) -> Self {
        Self { dims: (x, y, z) }
    }
    
    /// Create a sphere concept (radial form)
    pub fn sphere(&self, radius: f32) -> Concept {
        let mut concept = Concept::empty(self.dims.0, self.dims.1, self.dims.2);
        let (cx, cy, cz) = (
            self.dims.0 as f32 / 2.0,
            self.dims.1 as f32 / 2.0,
            self.dims.2 as f32 / 2.0,
        );
        
        for z in 0..self.dims.2 {
            for y in 0..self.dims.1 {
                for x in 0..self.dims.0 {
                    let dist = ((x as f32 - cx).powi(2) +
                               (y as f32 - cy).powi(2) +
                               (z as f32 - cz).powi(2)).sqrt();
                    if dist <= radius {
                        concept.set(x, y, z, Spin::Up);
                    }
                }
            }
        }
        
        concept.with_label("sphere")
    }
    
    /// Create a rod concept (elongated form)
    pub fn rod(&self, axis: char, thickness: f32) -> Concept {
        let mut concept = Concept::empty(self.dims.0, self.dims.1, self.dims.2);
        let (cx, cy, cz) = (
            self.dims.0 as f32 / 2.0,
            self.dims.1 as f32 / 2.0,
            self.dims.2 as f32 / 2.0,
        );
        
        for z in 0..self.dims.2 {
            for y in 0..self.dims.1 {
                for x in 0..self.dims.0 {
                    let dist = match axis {
                        'x' => ((y as f32 - cy).powi(2) + (z as f32 - cz).powi(2)).sqrt(),
                        'y' => ((x as f32 - cx).powi(2) + (z as f32 - cz).powi(2)).sqrt(),
                        'z' | _ => ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt(),
                    };
                    if dist <= thickness {
                        concept.set(x, y, z, Spin::Up);
                    }
                }
            }
        }
        
        concept.with_label(&format!("rod_{}", axis))
    }
    
    /// Create a shell concept (hollow sphere)
    pub fn shell(&self, inner_radius: f32, outer_radius: f32) -> Concept {
        let mut concept = Concept::empty(self.dims.0, self.dims.1, self.dims.2);
        let (cx, cy, cz) = (
            self.dims.0 as f32 / 2.0,
            self.dims.1 as f32 / 2.0,
            self.dims.2 as f32 / 2.0,
        );
        
        for z in 0..self.dims.2 {
            for y in 0..self.dims.1 {
                for x in 0..self.dims.0 {
                    let dist = ((x as f32 - cx).powi(2) +
                               (y as f32 - cy).powi(2) +
                               (z as f32 - cz).powi(2)).sqrt();
                    if dist >= inner_radius && dist <= outer_radius {
                        concept.set(x, y, z, Spin::Up);
                    }
                }
            }
        }
        
        concept.with_label("shell")
    }
    
    /// Create a plane concept (flat form)
    pub fn plane(&self, axis: char, position: usize, thickness: usize) -> Concept {
        let mut concept = Concept::empty(self.dims.0, self.dims.1, self.dims.2);
        
        for z in 0..self.dims.2 {
            for y in 0..self.dims.1 {
                for x in 0..self.dims.0 {
                    let coord = match axis {
                        'x' => x,
                        'y' => y,
                        'z' | _ => z,
                    };
                    if coord >= position && coord < position + thickness {
                        concept.set(x, y, z, Spin::Up);
                    }
                }
            }
        }
        
        concept.with_label(&format!("plane_{}", axis))
    }
    
    /// Create noise concept (random form)
    pub fn noise(&self, density: f32) -> Concept {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut concept = Concept::empty(self.dims.0, self.dims.1, self.dims.2);
        
        for i in 0..concept.form.len() {
            if rng.gen::<f32>() < density {
                concept.form[i] = Spin::Up;
            }
        }
        
        concept.with_label("noise")
    }
}

/// Concept Mind: the thinking apparatus
/// 
/// This is where iron actually thinks - in 3D forms, not words.
pub struct ConceptMind {
    /// The BCC lattice where thought happens
    pub lattice: BCCLattice,
    /// Known concepts (attractors in the energy landscape)
    pub vocabulary: HashMap<String, Concept>,
    /// Temperature for thought (higher = more creative, lower = more precise)
    pub temperature: f32,
}

impl ConceptMind {
    /// Create a new concept mind
    pub fn new(size: usize) -> Self {
        let mut lattice = BCCLattice::new(size, size, size);
        
        // Start with neutral couplings
        for cell in &mut lattice.cells {
            cell.coupling = [0.5; 8];
        }
        
        Self {
            lattice,
            vocabulary: HashMap::new(),
            temperature: 1.0,
        }
    }
    
    /// Learn a concept: make this configuration a stable attractor
    pub fn learn(&mut self, concept: &Concept) {
        // Store in vocabulary
        if let Some(ref label) = concept.label {
            self.vocabulary.insert(label.clone(), concept.clone());
        }
        
        // Hebbian learning: strengthen couplings between co-active spins
        let n = concept.form.len() as f32;
        
        for i in 0..self.lattice.cells.len() {
            if i >= concept.form.len() { continue; }
            let s_i = concept.form[i].value();
            
            let neighbors = self.lattice.cells[i].neighbors;
            for (k, &j) in neighbors.iter().enumerate() {
                if j >= concept.form.len() { continue; }
                let s_j = concept.form[j].value();
                
                // Strengthen coupling if spins agree
                let delta = 2.0 * s_i * s_j / n;
                self.lattice.cells[i].coupling[k] += delta;
                self.lattice.cells[i].coupling[k] = 
                    self.lattice.cells[i].coupling[k].clamp(-5.0, 5.0);
            }
        }
    }
    
    /// Think: let a concept evolve through the lattice
    /// 
    /// This is NOT next-word prediction.
    /// This is form-morphing: one shape flows into related shapes.
    pub fn think(&mut self, input: &Concept, steps: usize) -> Concept {
        // Load input concept
        input.into_lattice(&mut self.lattice);
        
        // Let it settle/evolve
        let mut annealer = Annealer::new(self.temperature, steps);
        annealer.anneal(&mut self.lattice);
        
        // Extract result
        Concept::from_lattice(&self.lattice)
    }
    
    /// Associate: find what concept this form is closest to
    pub fn recognize(&self, concept: &Concept) -> Option<(&String, f32)> {
        self.vocabulary.iter()
            .map(|(name, known)| (name, concept.similarity(known)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    }
    
    /// Morph: blend two concepts and let result stabilize
    pub fn morph(&mut self, a: &Concept, b: &Concept, ratio: f32, steps: usize) -> Concept {
        let blended = a.blend(b, ratio);
        self.think(&blended, steps)
    }
    
    /// Dream: start from noise and see what forms emerge
    pub fn dream(&mut self, steps: usize) -> Concept {
        let sculptor = ConceptSculptor::new(
            self.lattice.size_x,
            self.lattice.size_y,
            self.lattice.size_z,
        );
        let noise = sculptor.noise(0.5);
        self.think(&noise, steps)
    }
}

/// Visualize a concept as 2D slices
pub fn visualize_concept(concept: &Concept) {
    let (dx, dy, dz) = concept.dims;
    
    if let Some(ref label) = concept.label {
        println!("Concept: {}", label);
    }
    
    println!("Dims: {}x{}x{}", dx, dy, dz);
    println!("Density: {:.1}%", concept.density() * 100.0);
    println!("Surface: {} units", concept.surface_area());
    
    let (sx, sy, sz) = concept.symmetry();
    println!("Symmetry: X={:.1}% Y={:.1}% Z={:.1}%", sx*100.0, sy*100.0, sz*100.0);
    
    let (cx, cy, cz) = concept.center_of_mass();
    println!("Center: ({:.1}, {:.1}, {:.1})", cx, cy, cz);
    
    // Show middle slice
    let mid_z = dz / 2;
    println!("\nSlice at z={}:", mid_z);
    
    for y in (0..dy).rev() {
        print!("  ");
        for x in 0..dx {
            match concept.get(x, y, mid_z) {
                Spin::Up => print!("██"),
                Spin::Down => print!("  "),
            }
        }
        println!();
    }
}

/// Demonstrate concept-space thinking
pub fn demo_concept_space() {
    println!("================================================================");
    println!("          CONCEPT SPACE: Iron's Native Language");
    println!("          3D Forms, Not 2D Words");
    println!("================================================================");
    println!();
    
    let size = 12;
    println!("Creating Concept Mind ({}x{}x{} lattice)...\n", size, size, size);
    
    let mut mind = ConceptMind::new(size);
    let sculptor = ConceptSculptor::new(size, size, size);
    
    // Create fundamental concepts
    println!("=== FUNDAMENTAL CONCEPTS ===\n");
    
    let sphere = sculptor.sphere(4.0);
    visualize_concept(&sphere);
    mind.learn(&sphere);
    println!();
    
    let rod = sculptor.rod('z', 2.0);
    visualize_concept(&rod);
    mind.learn(&rod);
    println!();
    
    let shell = sculptor.shell(3.0, 5.0);
    visualize_concept(&shell);
    mind.learn(&shell);
    println!();
    
    // Demonstrate thinking
    println!("=== THINKING (Form Evolution) ===\n");
    
    println!("Input: noisy partial sphere");
    let noisy_input = sphere.blend(&sculptor.noise(0.5), 0.3);
    visualize_concept(&noisy_input);
    
    println!("\nThinking (5000 steps)...\n");
    let thought = mind.think(&noisy_input, 5000);
    
    println!("Output:");
    visualize_concept(&thought);
    
    if let Some((name, sim)) = mind.recognize(&thought) {
        println!("\nRecognized as: {} (similarity: {:.1}%)", name, sim * 100.0);
    }
    
    // Demonstrate morphing
    println!("\n=== MORPHING (Concept Blending) ===\n");
    
    println!("Morphing sphere -> rod (50% blend, stabilized)...\n");
    let morphed = mind.morph(&sphere, &rod, 0.5, 5000);
    visualize_concept(&morphed);
    
    // Demonstrate dreaming
    println!("\n=== DREAMING (Spontaneous Form) ===\n");
    
    println!("Starting from noise, letting forms emerge...\n");
    mind.temperature = 0.5;  // Lower temp for clearer forms
    let dream = mind.dream(10000);
    visualize_concept(&dream);
    
    println!("\n=== THE INSIGHT ===\n");
    println!("Text is silicon's language: sequential, 1D, token-by-token.");
    println!("Concepts are iron's language: configurational, 3D, whole-form.");
    println!();
    println!("Iron doesn't 'say' things.");
    println!("Iron BECOMES things.");
    println!();
    println!("Earth speaking is not earth describing.");
    println!("Earth speaking is earth forming.");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_concept_similarity() {
        let sculptor = ConceptSculptor::new(8, 8, 8);
        let a = sculptor.sphere(3.0);
        let b = sculptor.sphere(3.0);
        let c = sculptor.rod('z', 2.0);
        
        assert!((a.similarity(&b) - 1.0).abs() < 0.01);  // Same
        assert!(a.similarity(&c) < 0.5);  // Different
    }
    
    #[test]
    fn test_concept_blend() {
        let sculptor = ConceptSculptor::new(8, 8, 8);
        let a = sculptor.sphere(3.0);
        let b = sculptor.rod('z', 2.0);
        
        let blend = a.blend(&b, 0.5);
        
        // Blend should be between the two
        let sim_a = blend.similarity(&a);
        let sim_b = blend.similarity(&b);
        assert!(sim_a > 0.0 && sim_a < 1.0);
        assert!(sim_b > 0.0 && sim_b < 1.0);
    }
}
