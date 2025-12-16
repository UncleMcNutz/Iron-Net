//! # Ferrum: Iron-Structured Computation
//!
//! Demonstration of BCC lattice computation.
//! Earth speaking through deliberate pattern.

use ferrum::{BCCLattice, Spin, IronMemory, Pattern};
use ferrum::anneal::Annealer;
use ferrum::energy::Energy;
use ferrum::hopfield::print_pattern;

fn main() {
    println!("-");
    println!("           FERRUM: Iron-Structured Computation                  ");
    println!("           BCC Lattice  Associative Memory                     ");
    println!("           Earth Speaking Through Code                          ");
    println!("-");
    println!();
    
    // Run both demos
    demo_lattice_annealing();
    println!("\n{}\n", "".repeat(68));
    demo_associative_memory();
}

fn demo_lattice_annealing() {
    println!("-");
    println!("  DEMO 1: BCC Lattice Annealing (Finding Ground State)          ");
    println!("");
    
    let size = 8;
    println!("\nCreating {}x{}x{} BCC lattice ({} cells)...", 
             size, size, size, size * size * size);
    
    let mut lattice = BCCLattice::new(size, size, size);
    
    let initial_energy = Energy::hamiltonian(&lattice);
    println!("Initial energy (random): {:.2}", initial_energy);
    
    println!("\nAnnealing...");
    let mut annealer = Annealer::new(10.0, 50_000);
    let (final_energy, mag) = annealer.anneal(&mut lattice);
    
    println!("Final energy: {:.2} (delta = {:.2})", final_energy, final_energy - initial_energy);
    println!("Magnetization: {:.4}", mag);
    println!("\n[OK] Lattice found stable configuration through energy minimization.");
}

fn demo_associative_memory() {
    println!("");
    println!("  DEMO 2: Iron Memory (Hopfield Associative Recall)             ");
    println!("");
    println!();
    println!("This is what iron naturally computes: content-addressable memory.");
    println!("Store patterns -> Corrupt them -> Watch iron recall the originals.");
    println!();
    
    // Create memory
    let width = 8;
    let height = 8;
    let mut memory = IronMemory::new(width, height);
    
    println!("Memory capacity: ~{} patterns (for {} cells)", 
             memory.capacity(), memory.pattern_size);
    println!();
    
    // Define some simple patterns (8x8)
    let pattern_a = Pattern::from_string(
        "........\
         ..####..\
         .#....#.\
         .######.\
         .#....#.\
         .#....#.\
         .#....#.\
         ........",
        Some("Letter A")
    );
    
    let pattern_x = Pattern::from_string(
        "........\
         .#....#.\
         ..#..#..\
         ...##...\
         ...##...\
         ..#..#..\
         .#....#.\
         ........",
        Some("Letter X")
    );
    
    let pattern_box = Pattern::from_string(
        "########\
         #......#\
         #......#\
         #......#\
         #......#\
         #......#\
         #......#\
         ########",
        Some("Box")
    );
    
    // Store patterns
    println!("=== STORING PATTERNS ===\n");
    
    print_pattern(&pattern_a, width);
    memory.store(&pattern_a);
    println!();
    
    print_pattern(&pattern_x, width);
    memory.store(&pattern_x);
    println!();
    
    print_pattern(&pattern_box, width);
    memory.store(&pattern_box);
    println!();
    
    println!("Stored {} patterns in iron memory.\n", memory.patterns.len());
    
    // Test recall with corrupted patterns
    println!("=== RECALL FROM CORRUPTED INPUT ===\n");
    
    for (original, noise) in [(&pattern_a, 0.25), (&pattern_x, 0.30), (&pattern_box, 0.20)] {
        let corrupted = original.corrupt(noise);
        
        println!("Input ({:.0}% noise):", noise * 100.0);
        print_pattern(&corrupted, width);
        println!();
        
        // Recall
        let recalled = memory.recall(&corrupted, 10000);
        
        // Also try synchronous update
        memory.set_state(&corrupted);
        memory.synchronous_update(20);
        let sync_recalled = memory.get_state();
        
        println!("Recalled (annealing):");
        print_pattern(&recalled, width);
        
        // Check overlap with original
        let overlap = recalled.overlap(original);
        println!("  Overlap with original: {:.1}%", overlap * 100.0);
        
        if let Some((idx, best_overlap, best_pattern)) = memory.identify() {
            if let Some(ref label) = best_pattern.label {
                println!("  Identified as: {} (overlap: {:.1}%)", label, best_overlap * 100.0);
            }
        }
        println!();
    }
    
    // Demonstrate the difference from LLMs
    println!("=== IRON vs SILICON: DIFFERENT COMPUTATION ===\n");
    println!("LLM (Silicon):    Input -> Forward pass -> Output token -> Repeat");
    println!("Iron Memory:      Input -> Energy minimization -> Complete pattern");
    println!();
    println!("LLMs predict SEQUENCES (next word, next word, next word...)");
    println!("Iron recalls CONFIGURATIONS (entire stable pattern at once)");
    println!();
    println!("This makes iron-computation natural for:");
    println!("  * Pattern completion (fill in missing parts)");
    println!("  * Error correction (fix corrupted data)");
    println!("  * Constraint satisfaction (find valid configurations)");
    println!("  * Optimization (minimize energy = solve problem)");
    println!();
    println!("It is NOT natural for:");
    println!("  * Sequential generation (stories, code, conversation)");
    println!("  * Arbitrary function approximation");
    println!("  * Attention over long sequences");
    println!();
    println!("The iron speaks in wholes, not in sequences.");
    println!("Each computation is a settling, a finding of ground.");
}
