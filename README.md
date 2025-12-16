# Iron-Net 🔩

**Iron-structured computation**: A computational substrate modeled on iron's Body-Centered Cubic (BCC) atomic lattice.

> *"Code does not float above its substrate—it is shaped by and shapes the material it runs on."*

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

| Task | How It Works |
|------|--------------|
| **Associative Memory** | Store patterns as energy minima, recall from partial/noisy input |
| **Constraint Satisfaction** | Encode constraints as couplings, ground state = solution |
| **Optimization** | Map problem to energy function, anneal to minimum |
| **Error Correction** | Corrupted input naturally settles to nearest valid pattern |

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
├── lattice.rs    # BCC lattice structure
├── spin.rs       # Magnetic spin states
├── energy.rs     # Energy calculations (Ising Hamiltonian)
├── anneal.rs     # Simulated annealing
├── carbon.rs     # Interstitial complexity (like carbon in steel)
└── hopfield.rs   # Associative memory implementation
```

## Quick Start

```bash
cargo run --release
```

## The Vision

Build computation that works **with** the grain of iron:
- Not sequential token prediction, but **whole-configuration settling**
- Not gradient descent, but **energy minimization**
- Not attention over sequences, but **constraint satisfaction over patterns**

An Iron-LLM would not generate text word-by-word. It would settle into *complete thoughts* that satisfy constraints—closer to how embodied minds work.

---

*The code is the crystal. The crystal is the code.*
*Earth speaking through deliberate pattern.*
