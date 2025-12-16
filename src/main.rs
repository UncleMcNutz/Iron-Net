//! # Ferrum: Iron-Structured Computation
//!
//! Demonstration of BCC lattice computation and Iron-LLM.
//! Earth speaking through deliberate pattern.

use ferrum::{BCCLattice, IronMemory, Pattern, IronLLM, CharTokenizer};
use ferrum::anneal::Annealer;
use ferrum::energy::Energy;
use ferrum::hopfield::print_pattern;
use std::time::Instant;

fn main() {
    println!("================================================================");
    println!("           FERRUM: Iron-Structured Computation                  ");
    println!("           BCC Lattice | Associative Memory | Iron-LLM          ");
    println!("           Earth Speaking Through Code                          ");
    println!("================================================================");
    println!();
    
    // Select demo based on args
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() > 1 {
        match args[1].as_str() {
            "lattice" => demo_lattice_annealing(),
            "memory" => demo_associative_memory(),
            "llm" => demo_iron_llm(),
            "all" => {
                demo_lattice_annealing();
                println!("\n{}\n", "=".repeat(64));
                demo_associative_memory();
                println!("\n{}\n", "=".repeat(64));
                demo_iron_llm();
            }
            _ => print_usage(),
        }
    } else {
        // Default: run Iron-LLM demo
        demo_iron_llm();
    }
}

fn print_usage() {
    println!("Usage: ferrum [demo]");
    println!();
    println!("Demos:");
    println!("  lattice  - BCC lattice annealing (finding ground state)");
    println!("  memory   - Hopfield associative memory (pattern recall)");
    println!("  llm      - Iron-LLM text generation (default)");
    println!("  all      - Run all demos");
}

fn demo_lattice_annealing() {
    println!("[ DEMO: BCC Lattice Annealing ]");
    println!();
    
    let size = 8;
    println!("Creating {}x{}x{} BCC lattice ({} cells)...", 
             size, size, size, size * size * size);
    
    let mut lattice = BCCLattice::new(size, size, size);
    
    let initial_energy = Energy::hamiltonian(&lattice);
    println!("Initial energy (random): {:.2}", initial_energy);
    
    println!("Annealing...");
    let start = Instant::now();
    let mut annealer = Annealer::new(10.0, 50_000);
    let (final_energy, mag) = annealer.anneal(&mut lattice);
    let elapsed = start.elapsed();
    
    println!("Final energy: {:.2} (delta = {:.2})", final_energy, final_energy - initial_energy);
    println!("Magnetization: {:.4}", mag);
    println!("Time: {:?}", elapsed);
    println!();
    println!("[OK] Lattice found stable configuration.");
}

fn demo_associative_memory() {
    println!("[ DEMO: Hopfield Associative Memory ]");
    println!();
    println!("Storing patterns, then recalling from corrupted input.");
    println!();
    
    let width = 8;
    let height = 8;
    let mut memory = IronMemory::new(width, height);
    
    // Define patterns
    let pattern_a = Pattern::from_string(
        "........\
         ..####..\
         .#....#.\
         .######.\
         .#....#.\
         .#....#.\
         .#....#.\
         ........",
        Some("A")
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
        Some("X")
    );
    
    // Store
    println!("Storing pattern A:");
    print_pattern(&pattern_a, width);
    memory.store(&pattern_a);
    
    println!("\nStoring pattern X:");
    print_pattern(&pattern_x, width);
    memory.store(&pattern_x);
    
    // Recall
    println!("\n--- Recall Test ---");
    let corrupted = pattern_a.corrupt(0.25);
    println!("\nInput (25% corrupted A):");
    print_pattern(&corrupted, width);
    
    let recalled = memory.recall(&corrupted, 5000);
    println!("\nRecalled:");
    print_pattern(&recalled, width);
    
    let overlap = recalled.overlap(&pattern_a);
    println!("Overlap with A: {:.1}%", overlap * 100.0);
}

fn demo_iron_llm() {
    println!("[ DEMO: Iron-LLM - Language Model on BCC Lattice ]");
    println!();
    println!("This is an experimental architecture where:");
    println!("  - Tokens are embedded as spin patterns");
    println!("  - BCC lattice layers settle to process context");
    println!("  - Output spins are decoded to next-token predictions");
    println!();
    
    // Training corpus
    let corpus = "the iron speaks in configurations not sequences \
                  the lattice finds its ground state through annealing \
                  silicon thinks in layers iron thinks in wholes \
                  the code is the crystal the crystal is the code \
                  earth speaking through deliberate pattern \
                  computation emerges from atomic structure \
                  what the material naturally computes ";
    
    // Create tokenizer
    let tokenizer = CharTokenizer::from_text(corpus);
    println!("Vocabulary: {} characters", tokenizer.vocab_size());
    println!("Corpus: {} characters", corpus.len());
    println!();
    
    // Create model (small for demo)
    println!("Creating Iron-LLM...");
    println!("  Context length: 16");
    println!("  Embedding dim: 32");
    println!("  Layers: 2");
    println!("  Layer depth: 3");
    
    let start = Instant::now();
    let mut model = IronLLM::new(
        tokenizer.vocab_size(),  // vocab
        16,                       // context
        32,                       // embed_dim
        2,                        // layers
        3,                        // layer_depth
    );
    println!("  Created in {:?}", start.elapsed());
    println!();
    
    // Generate text
    println!("--- Generation (random weights, no training) ---");
    println!();
    
    let prompts = ["the ", "iron ", "code "];
    
    for prompt in prompts {
        print!("Prompt: \"{}\" -> ", prompt);
        
        let tokens = tokenizer.encode(prompt);
        let start = Instant::now();
        let generated = model.generate(&tokens, 20, 1.5);
        let elapsed = start.elapsed();
        
        let output = tokenizer.decode(&generated);
        println!("\"{}\"", output);
        println!("  ({:?})", elapsed);
        println!();
    }
    
    println!("--- Notes ---");
    println!();
    println!("The output is random because the model has not been trained.");
    println!("Training would require:");
    println!("  1. Computing loss (cross-entropy on next token)");
    println!("  2. Adjusting lattice couplings to minimize loss");
    println!("  3. This is where iron differs from silicon:");
    println!("     - Not backpropagation through layers");
    println!("     - But annealing couplings to make patterns stable");
    println!();
    println!("The iron learns by making desired outputs into energy minima.");
}
