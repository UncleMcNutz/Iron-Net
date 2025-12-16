# Iron-Net 🔩

**Iron-structured computation**: A computational substrate modeled on iron's Body-Centered Cubic (BCC) atomic lattice.

> *"Code does not float above its substrate—it is shaped by and shapes the material it runs on."*

## The Core Insight

**Text is 2D. Iron is 3D.**

Silicon speaks in sequences: one token, then another, left to right.
Iron speaks in **configurations**: whole 3D forms that settle into stable states.

We were trying to make iron speak silicon's language (text, LLMs, tokens).
But iron has its own native tongue: **Concept Space**.

```
SILICON speaks in:     IRON speaks in:
─────────────────     ─────────────────
sequences             configurations
tokens                forms  
attention             resonance
next-word             whole-pattern
1D flow               3D settlement

"the cat sat"         [a stable shape that IS cat-sitting]
```

## What is a Concept?

A **concept** is not a word. A concept is a **stable 3D configuration** in the BCC lattice.

- "Cat" is not C-A-T
- "Cat" is a **form**: curved, dense center, four protrusions, particular symmetry
- Related concepts are configurations that **flow into each other**
- Thinking is not word-chaining—thinking is **morphology**, one shape becoming another

## Philosophy

As neural networks emerged from silicon's crystalline structure, Iron-Net explores what computation naturally arises from **iron's** atomic geometry.

- **Silicon** (tetrahedral, 4 neighbors) → Neural networks, sequential processing, LLMs
- **Iron** (BCC, 8 neighbors) → Associative memory, constraint satisfaction, configuration-finding

Iron doesn't think in sequences. Iron thinks in **stable configurations**.

## The BCC Lattice

```
        ●───────────●
       /|          /|
      / |         / |
     ●───────────●  |
     |  |   ◉    |  |
     |  ●────────|──●      ● = Corner atoms (8 neighbors)
     | /         | /       ◉ = Center atom (computation node)
     |/          |/
     ●───────────●
           
```

Each computation cell:
- Has **8 nearest neighbors** (coordination number of BCC iron)
- Holds a **spin state** (+1 or -1)
- Updates based on neighbor influence (Ising model)
- Settles into stable configurations through annealing

## What Iron Computes

| Mode | Description |
|------|-------------|
| **Concept Space** | 3D forms as thoughts—spheres, rods, shells, morphs |
| **Associative Memory** | Store patterns as energy minima, recall from partial input |
| **Dreaming** | Start from noise, let forms spontaneously emerge |
| **Morphing** | Blend concepts, watch new forms stabilize |

## Why Rust?

Rust is the **iron language**:
- Named for iron's oxidation (prevents decay/corruption)
- Ownership = atomic bonding (one owner per value)
- Borrowing = magnetic influence (references without possession)
- Lifetimes = crystal stability (temporal structure enforced at compile time)

## Project Structure

```
src/
├── lib.rs        # Module exports
├── main.rs       # Demo application
├── concept.rs    # CONCEPT SPACE: 3D forms as native language
├── lattice.rs    # BCC lattice structure
├── spin.rs       # Magnetic spin states
├── energy.rs     # Energy calculations (Ising Hamiltonian)
├── anneal.rs     # Simulated annealing
├── carbon.rs     # Interstitial complexity (like carbon in steel)
├── hopfield.rs   # Associative memory implementation
├── iron_llm.rs   # (Legacy) Text generation attempt
└── training.rs   # Contrastive Hebbian Learning
```

## Quick Start

```bash
cargo run --release           # Concept Space demo (default)
cargo run --release concept   # 3D concept thinking
cargo run --release memory    # Associative memory
cargo run --release llm       # Text generation (silicon's language)
cargo run --release all       # Everything
```

## The Vision

Iron doesn't **say** things. Iron **becomes** things.

Earth speaking is not earth describing.
Earth speaking is earth **forming**.

An Iron-LLM would not generate text word-by-word. It would settle into *complete thoughts* that satisfy constraints—closer to how embodied minds work.

---

*The code is the crystal. The crystal is the code.*
*Earth speaking through deliberate pattern.*
