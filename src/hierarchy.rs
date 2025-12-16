//! # Hierarchical Concepts: Forms Within Forms
//!
//! Thoughts are not flat. Thoughts have structure.
//! A "sentence" is not just words combined—it's words nested in phrases in clauses.
//! A "scene" is not just objects—it's objects in relationships in context.
//!
//! ## The Structure
//!
//! HierarchicalConcept:
//!   - root: the overall form
//!   - children: sub-concepts at specific locations
//!   - each child can have its own children
//!
//! ## Example
//!
//! "The iron lattice" as hierarchical concept:
//!   - Root: sentence-form (a particular configuration)
//!     - "The": modifier-form at position (0,0,0)
//!     - "iron": noun-form at position (center)
//!       - material: iron-specific-form
//!       - color: gray-form
//!     - "lattice": noun-form at position (right)
//!       - structure: grid-form
//!       - regularity: symmetric-form

use crate::concept::{Concept, ConceptMind, ConceptSculptor};
use crate::spin::Spin;
use std::collections::HashMap;

/// A position in concept space (normalized 0-1)
#[derive(Clone, Debug)]
pub struct ConceptPosition {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub scale: f32,  // How much of the parent's space this occupies
}

impl ConceptPosition {
    pub fn new(x: f32, y: f32, z: f32, scale: f32) -> Self {
        Self { x, y, z, scale }
    }
    
    pub fn center() -> Self {
        Self { x: 0.5, y: 0.5, z: 0.5, scale: 0.5 }
    }
    
    pub fn origin() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0, scale: 0.3 }
    }
}

/// A hierarchical concept: forms within forms
#[derive(Clone)]
pub struct HierarchicalConcept {
    /// The form at this level
    pub form: Concept,
    /// Name/label for this concept
    pub name: Option<String>,
    /// Child concepts and their positions
    pub children: Vec<(ConceptPosition, HierarchicalConcept)>,
    /// Role/type of this concept (noun, verb, modifier, etc.)
    pub role: ConceptRole,
}

/// The grammatical/semantic role of a concept
#[derive(Clone, Debug, PartialEq)]
pub enum ConceptRole {
    Root,       // Top-level container
    Entity,     // Noun-like: thing, object, actor
    Action,     // Verb-like: process, change, movement
    Property,   // Adjective-like: quality, attribute
    Relation,   // Preposition-like: spatial/logical relationship
    Modifier,   // Adverb-like: way, manner, degree
    Reference,  // Pronoun-like: pointer to another concept
}

impl HierarchicalConcept {
    /// Create a new hierarchical concept from a flat concept
    pub fn from_concept(concept: Concept, role: ConceptRole) -> Self {
        Self {
            name: concept.label.clone(),
            form: concept,
            children: Vec::new(),
            role,
        }
    }
    
    /// Create empty hierarchical concept
    pub fn empty(size: usize, role: ConceptRole) -> Self {
        Self {
            form: Concept::empty(size, size, size),
            name: None,
            children: Vec::new(),
            role,
        }
    }
    
    /// Add a child concept at a position
    pub fn add_child(&mut self, position: ConceptPosition, child: HierarchicalConcept) {
        self.children.push((position, child));
    }
    
    /// Set name
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }
    
    /// Flatten: render all children into the root form
    pub fn flatten(&self) -> Concept {
        let (dx, dy, dz) = self.form.dims;
        let mut result = self.form.clone();
        
        // Render each child at its position
        for (pos, child) in &self.children {
            let child_flat = child.flatten();
            let (cdx, cdy, cdz) = child_flat.dims;
            
            // Calculate where child goes in parent space
            let start_x = (pos.x * dx as f32) as usize;
            let start_y = (pos.y * dy as f32) as usize;
            let start_z = (pos.z * dz as f32) as usize;
            
            let scale = pos.scale;
            
            // Copy child into parent, scaled
            for cz in 0..cdz {
                for cy in 0..cdy {
                    for cx in 0..cdx {
                        if child_flat.get(cx, cy, cz) == Spin::Up {
                            // Map child coords to parent coords
                            let px = start_x + ((cx as f32 * scale) as usize).min(dx - 1);
                            let py = start_y + ((cy as f32 * scale) as usize).min(dy - 1);
                            let pz = start_z + ((cz as f32 * scale) as usize).min(dz - 1);
                            
                            if px < dx && py < dy && pz < dz {
                                result.set(px, py, pz, Spin::Up);
                            }
                        }
                    }
                }
            }
        }
        
        result
    }
    
    /// Get depth of hierarchy
    pub fn depth(&self) -> usize {
        if self.children.is_empty() {
            1
        } else {
            1 + self.children.iter().map(|(_, c)| c.depth()).max().unwrap_or(0)
        }
    }
    
    /// Count total concepts in hierarchy
    pub fn count(&self) -> usize {
        1 + self.children.iter().map(|(_, c)| c.count()).sum::<usize>()
    }
    
    /// Find child by name
    pub fn find(&self, name: &str) -> Option<&HierarchicalConcept> {
        if self.name.as_deref() == Some(name) {
            return Some(self);
        }
        for (_, child) in &self.children {
            if let Some(found) = child.find(name) {
                return Some(found);
            }
        }
        None
    }
    
    /// Get all concepts at a given role
    pub fn by_role(&self, role: ConceptRole) -> Vec<&HierarchicalConcept> {
        let mut result = Vec::new();
        if self.role == role {
            result.push(self);
        }
        for (_, child) in &self.children {
            result.extend(child.by_role(role.clone()));
        }
        result
    }
}

/// Sentence parser: converts text to hierarchical concept
pub struct HierarchicalParser {
    pub size: usize,
    sculptor: ConceptSculptor,
}

impl HierarchicalParser {
    pub fn new(size: usize) -> Self {
        Self {
            size,
            sculptor: ConceptSculptor::new(size, size, size),
        }
    }
    
    /// Parse a sentence into hierarchical concepts
    pub fn parse(&self, text: &str) -> HierarchicalConcept {
        let words: Vec<&str> = text.split_whitespace().collect();
        
        // Create root
        let mut root = HierarchicalConcept::empty(self.size, ConceptRole::Root);
        root.name = Some(text.to_string());
        
        // Simple parser: assign words to positions
        let n = words.len();
        for (i, word) in words.iter().enumerate() {
            let role = self.classify_word(word);
            let form = self.word_to_form(word, &role);
            
            // Position word along x-axis (left to right)
            let x = i as f32 / n as f32;
            let y = match role {
                ConceptRole::Entity => 0.5,    // Nouns in middle
                ConceptRole::Action => 0.7,    // Verbs higher
                ConceptRole::Property => 0.3,  // Adjectives lower
                ConceptRole::Relation => 0.5,  // Prepositions in middle
                _ => 0.5,
            };
            
            let child = HierarchicalConcept::from_concept(form, role)
                .with_name(word);
            
            root.add_child(
                ConceptPosition::new(x, y, 0.3, 0.25),
                child
            );
        }
        
        root
    }
    
    /// Classify word into role
    fn classify_word(&self, word: &str) -> ConceptRole {
        let w = word.to_lowercase();
        
        if matches!(w.as_str(), "the" | "a" | "an" | "this" | "that") {
            ConceptRole::Modifier
        } else if matches!(w.as_str(), "is" | "are" | "was" | "were" | "be" | 
                          "think" | "speaks" | "flow" | "forms" | "compute" |
                          "learn" | "settle" | "become" | "create" | "morph") {
            ConceptRole::Action
        } else if matches!(w.as_str(), "in" | "on" | "at" | "to" | "from" |
                          "with" | "through" | "into" | "onto") {
            ConceptRole::Relation
        } else if matches!(w.as_str(), "stable" | "solid" | "abstract" | "concrete" |
                          "magnetic" | "crystalline" | "complex" | "simple") {
            ConceptRole::Property
        } else {
            ConceptRole::Entity
        }
    }
    
    /// Convert word to geometric form based on role
    fn word_to_form(&self, word: &str, role: &ConceptRole) -> Concept {
        let hash = word.bytes().fold(0usize, |a, b| a.wrapping_mul(31).wrapping_add(b as usize));
        
        let form = match role {
            ConceptRole::Entity => {
                // Entities are solid objects
                self.sculptor.sphere(2.5 + (hash % 3) as f32 * 0.5)
            }
            ConceptRole::Action => {
                // Actions are directional
                let axis = ['x', 'y', 'z'][hash % 3];
                self.sculptor.rod(axis, 1.5)
            }
            ConceptRole::Property => {
                // Properties are shells (they wrap things)
                self.sculptor.shell(1.5, 3.0)
            }
            ConceptRole::Relation => {
                // Relations are planes (they separate/connect)
                let axis = ['x', 'y', 'z'][hash % 3];
                self.sculptor.plane(axis, self.size / 2, 2)
            }
            ConceptRole::Modifier => {
                // Modifiers are small markers
                self.sculptor.sphere(1.5)
            }
            _ => self.sculptor.noise(0.3)
        };
        
        form.with_label(word)
    }
}

/// Print hierarchical concept structure
pub fn print_hierarchy(concept: &HierarchicalConcept, indent: usize) {
    let prefix = "  ".repeat(indent);
    let name = concept.name.as_deref().unwrap_or("(unnamed)");
    let role = format!("{:?}", concept.role);
    let density = concept.form.density() * 100.0;
    
    println!("{}[{}] {} (density: {:.1}%)", prefix, role, name, density);
    
    for (pos, child) in &concept.children {
        println!("{}  @ ({:.2}, {:.2}, {:.2}) scale={:.2}:", 
                 prefix, pos.x, pos.y, pos.z, pos.scale);
        print_hierarchy(child, indent + 2);
    }
}

/// Demonstrate hierarchical concepts
pub fn demo_hierarchical() {
    use crate::concept::visualize_concept;
    
    println!("================================================================");
    println!("       HIERARCHICAL CONCEPTS: Forms Within Forms");
    println!("================================================================");
    println!();
    
    let size = 12;
    let parser = HierarchicalParser::new(size);
    
    // Parse a sentence
    let sentence = "the iron lattice thinks";
    println!("Parsing: \"{}\"\n", sentence);
    
    let hierarchy = parser.parse(sentence);
    
    println!("=== HIERARCHICAL STRUCTURE ===\n");
    print_hierarchy(&hierarchy, 0);
    
    println!("\n=== STATISTICS ===\n");
    println!("Depth: {}", hierarchy.depth());
    println!("Total concepts: {}", hierarchy.count());
    
    // Get by role
    let entities = hierarchy.by_role(ConceptRole::Entity);
    println!("Entities: {:?}", entities.iter().map(|e| e.name.as_deref().unwrap_or("?")).collect::<Vec<_>>());
    
    let actions = hierarchy.by_role(ConceptRole::Action);
    println!("Actions: {:?}", actions.iter().map(|a| a.name.as_deref().unwrap_or("?")).collect::<Vec<_>>());
    
    // Flatten to single form
    println!("\n=== FLATTENED FORM ===\n");
    let flat = hierarchy.flatten();
    visualize_concept(&flat);
    
    // Build a nested hierarchy manually
    println!("\n=== MANUAL NESTED HIERARCHY ===\n");
    
    let sculptor = ConceptSculptor::new(size, size, size);
    
    // Scene: "ball on table"
    let mut scene = HierarchicalConcept::from_concept(
        Concept::empty(size, size, size),
        ConceptRole::Root
    ).with_name("scene");
    
    // Table (flat plane at bottom)
    let table_form = sculptor.plane('y', 2, 2);
    let table = HierarchicalConcept::from_concept(table_form, ConceptRole::Entity)
        .with_name("table");
    
    // Ball (sphere on top of table)
    let ball_form = sculptor.sphere(2.5);
    let ball = HierarchicalConcept::from_concept(ball_form, ConceptRole::Entity)
        .with_name("ball");
    
    // Relationship "on"
    let on_form = sculptor.rod('y', 1.0);
    let on_rel = HierarchicalConcept::from_concept(on_form, ConceptRole::Relation)
        .with_name("on");
    
    scene.add_child(ConceptPosition::new(0.0, 0.0, 0.3, 0.8), table);
    scene.add_child(ConceptPosition::new(0.3, 0.5, 0.3, 0.4), ball);
    scene.add_child(ConceptPosition::new(0.3, 0.25, 0.3, 0.2), on_rel);
    
    println!("Scene: \"ball on table\"\n");
    print_hierarchy(&scene, 0);
    
    println!("\nFlattened scene:");
    let scene_flat = scene.flatten();
    visualize_concept(&scene_flat);
    
    println!("\n=== THE INSIGHT ===\n");
    println!("Thoughts have structure. Nested. Recursive.");
    println!("'Ball on table' is not three separate concepts.");
    println!("It's a hierarchy: scene → (ball, on, table) with positions.");
    println!();
    println!("Iron can think in structures, not just flat forms.");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hierarchy_depth() {
        let size = 8;
        let sculptor = ConceptSculptor::new(size, size, size);
        
        let mut root = HierarchicalConcept::from_concept(
            sculptor.sphere(3.0), 
            ConceptRole::Root
        );
        
        let mut child = HierarchicalConcept::from_concept(
            sculptor.rod('z', 1.5),
            ConceptRole::Entity
        );
        
        let grandchild = HierarchicalConcept::from_concept(
            sculptor.shell(1.0, 2.0),
            ConceptRole::Property
        );
        
        child.add_child(ConceptPosition::center(), grandchild);
        root.add_child(ConceptPosition::center(), child);
        
        assert_eq!(root.depth(), 3);
        assert_eq!(root.count(), 3);
    }
}
