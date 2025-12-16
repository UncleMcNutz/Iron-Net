//! # Ferrum: Iron-Structured Computation
//!
//! Demonstration of BCC lattice computation and Iron-LLM.
//! Earth speaking through deliberate pattern.

use ferrum::{BCCLattice, IronMemory, Pattern, IronLLM, CharTokenizer};
use ferrum::anneal::Annealer;
use ferrum::energy::Energy;
use ferrum::hopfield::print_pattern;
use ferrum::training::{IronTrainer, TrainConfig, demo_training};
use ferrum::concept::demo_concept_space;
use std::time::Instant;

fn main() {
    println!("================================================================");
    println!("           FERRUM: Iron-Structured Computation                  ");
    println!("           BCC Lattice | Associative Memory | Iron-LLM          ");
    println!("           Earth Speaking Through Code                          ");
    println!("================================================================");
    println!();
    
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() > 1 {
        match args[1].as_str() {
            "lattice" => demo_lattice_annealing(),
            "memory" => demo_associative_memory(),
            "llm" => demo_iron_llm(),
            "train" => demo_iron_llm_training(),
            "concept" => demo_concept_space(),  // The main event
            "all" => {
                demo_lattice_annealing();
                println!("\n{}\n", "=".repeat(64));
                demo_associative_memory();
                println!("\n{}\n", "=".repeat(64));
                demo_iron_llm();
                println!("\n{}\n", "=".repeat(64));
                demo_iron_llm_training();
                println!("\n{}\n", "=".repeat(64));
                demo_concept_space();
            }
            _ => print_usage(),
        }
    } else {
        demo_concept_space();  // Default to the main mode
    }
}

fn print_usage() {
    println!("Usage: ferrum [demo]");
    println!();
    println!("Demos:");
    println!("  concept  - 3D concept space (default) - Iron's native language");
    println!("  lattice  - BCC lattice annealing (finding ground state)");
    println!("  memory   - Hopfield associative memory (pattern recall)");
    println!("  llm      - Iron-LLM text generation (silicon's language)");
    println!("  train    - Train Iron-LLM on sample text");
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
    
    let width = 8;
    let height = 8;
    let mut memory = IronMemory::new(width, height);
    
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
    
    println!("Storing pattern A:");
    print_pattern(&pattern_a, width);
    memory.store(&pattern_a);
    
    println!("\nStoring pattern X:");
    print_pattern(&pattern_x, width);
    memory.store(&pattern_x);
    
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
    
    let corpus = "the iron speaks in configurations not sequences \
                  the lattice finds its ground state through annealing \
                  silicon thinks in layers iron thinks in wholes \
                  the code is the crystal the crystal is the code ";
    
    let tokenizer = CharTokenizer::from_text(corpus);
    println!("Vocabulary: {} chars", tokenizer.vocab_size());
    
    println!("Creating Iron-LLM (16 context, 32 dim, 2 layers)...");
    let mut model = IronLLM::new(tokenizer.vocab_size(), 16, 32, 2, 3);
    
    println!("\n--- Generation (untrained) ---\n");
    
    for prompt in ["the ", "iron ", "code "] {
        let tokens = tokenizer.encode(prompt);
        let generated = model.generate(&tokens, 20, 1.5);
        let output = tokenizer.decode(&generated);
        println!("\"{}\" -> \"{}\"", prompt.trim(), output);
    }
}

fn demo_iron_llm_training() {
    println!("[ DEMO: Training Iron-LLM ]");
    println!();
    println!("Training method: Contrastive Hebbian Learning");
    println!("  - Clamped phase: Input + correct output fixed, middle settles");
    println!("  - Free phase: Only input fixed, network settles freely");  
    println!("  - Update: Strengthen couplings that differ between phases");
    println!();
    
    // Smaller corpus for demo
    let corpus = "the iron the code the iron the code the lattice the crystal ";
    
    let tokenizer = CharTokenizer::from_text(corpus);
    println!("Vocabulary: {} chars", tokenizer.vocab_size());
    println!("Corpus: \"{}\"", corpus.trim());
    println!();
    
    // Tiny model for fast training demo
    println!("Creating tiny Iron-LLM (8 context, 16 dim, 1 layer)...");
    let mut model = IronLLM::new(
        tokenizer.vocab_size(),
        8,    // context
        16,   // embed_dim
        1,    // layers
        2,    // layer_depth
    );
    
    // Before training
    println!("\n--- Before Training ---");
    let tokens = tokenizer.encode("the ");
    let generated = model.generate(&tokens, 15, 1.0);
    println!("\"the \" -> \"{}\"", tokenizer.decode(&generated));
    
    // Train
    println!("\n--- Training (3 epochs) ---\n");
    demo_training(&mut model, &tokenizer, corpus, 3);
    
    // After training
    println!("--- After Training ---");
    let tokens = tokenizer.encode("the ");
    let generated = model.generate(&tokens, 15, 1.0);
    println!("\"the \" -> \"{}\"", tokenizer.decode(&generated));
    
    let tokens = tokenizer.encode("iron ");
    let generated = model.generate(&tokens, 15, 1.0);
    println!("\"iron \" -> \"{}\"", tokenizer.decode(&generated));
    
    println!();
    println!("Note: With more training data and larger model, patterns emerge.");
    println!("The iron learns by making correct sequences into energy minima.");
}
