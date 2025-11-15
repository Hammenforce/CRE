# CRE: Complex Resonance Embedding

[![License: Academic Free / Commercial](https://img.shields.io/badge/License-Academic%20Free%20%2F%20Commercial-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

A novel sequence processing architecture achieving **O(n) complexity** through push-pull cumulative dynamics and position-modulated resonance.

## Key Results

Comprehensive benchmarking with matched parameters (~4.76M vs ~4.74M, ratio: 1.005) on NVIDIA RTX 5090:

| Metric | CRE | Transformer | Flash Attention |
|--------|-----|-------------|-----------------|
| **Runtime @ L=32768** | 142.7ms | OOM | 302.6ms |
| **Memory @ L=16384** | 266.5MB | 32,943.5MB | 319.5MB |
| **Bracket AUC (avg)** | 0.900 | 0.925 | 0.920 |
| **ECL-90 @ L=4096** | 2389.3 | 2389.3 | 2389.3 |
| **Speedup @ L=16384** | **21.7x** vs Trans | - | **2.5x** vs Flash |

### Highlights

- **O(n) Memory Scaling**: Empirically validated linear growth (467MB at L=32768)
- **21.7x Faster** than standard Transformer at L=16384
- **2.1x Faster** than Flash Attention at L=32768
- **124x Less Memory** than Transformer at L=16384
- **Competitive Quality**: 0.900 AUC vs 0.925 (2.5% difference)
- **Identical Context Length**: ECL-90 matches Transformer exactly across all tested lengths
- **Processes 32K tokens** where standard Transformer fails due to OOM

## Architecture Overview

CRE replaces quadratic attention with linear cumulative operations:

```
Standard Attention: O(n²) - computes all pairwise interactions
CRE:               O(n)  - cumulative push-pull with position modulation
```

### Core Mechanism

```python
# Push-Pull Dynamics with Position Modulation
g_t = sigmoid(W_g * x_t)           # Gate controls information routing
v_t = W_v * x_t                    # Value projection

P_t = cumsum(g * v)                # Push accumulator
Q_t = cumsum((1-g) * v)            # Pull accumulator

h_t = exp(-λt) * (P_t - Q_t)       # Position-modulated resonance output

# Where:
# λ is a learnable decay rate
# g_t ∈ (0,1) controls selective accumulation
# The differential (P_t - Q_t) captures the resonance pattern
```

### Key Innovations

1. **Push-Pull Dynamics**: Differential accumulation enables selective information retention through complementary accumulators
2. **Position Modulation**: Learnable exponential weighting normalizes cumulative sums and creates adaptive position-dependent patterns
3. **Multi-Frequency Resonance**: Multiple modulation rates capture both short-term and long-term patterns
4. **Gated Information Flow**: Learnable gates control what information enters each accumulator
5. **Linear Complexity**: All operations are O(n) in sequence length

## Installation

```bash
git clone https://github.com/Hammenforce/CRE.git
cd CRE
pip install torch numpy scipy scikit-learn matplotlib
```

Requirements:
- Python 3.8+
- PyTorch 2.0+
- NumPy
- SciPy
- scikit-learn
- Matplotlib

## Quick Start

### Basic Usage

```python
from cre import CRELayer, CREEncoder

# Single layer
layer = CRELayer(d_model=256, n_frequencies=3)
x = torch.randn(batch_size, seq_length, 256)
output = layer(x)

# Full encoder
encoder = CREEncoder(
    vocab_size=10000,
    d_model=256,
    n_layers=6,
    n_frequencies=3
)
logits = encoder(input_ids)
```

### Sequence Classification

```python
from cre import CREForSequenceClassification

model = CREForSequenceClassification(
    vocab_size=10000,
    d_model=256,
    n_classes=2,
    n_layers=6
)

input_ids = torch.randint(0, 10000, (batch_size, seq_len))
logits = model(input_ids)
```

### Language Modeling

```python
from cre import CREForLanguageModeling

model = CREForLanguageModeling(
    vocab_size=10000,
    d_model=256,
    n_layers=6,
    tie_weights=True
)

input_ids = torch.randint(0, 10000, (batch_size, seq_len))
logits = model(input_ids)  # (batch, seq_len, vocab_size)
```

## Benchmarking

### Full Benchmark Suite

```bash
python scientific_benchmark.py --seeds 5 --max-seq-len 32768
```

This evaluates:
- Runtime scaling (O(n) verification)
- Memory consumption
- Effective Context Length (ECL)
- Structural reasoning (bracket matching)
- Ablation studies

### Quick Test

```bash
python scientific_benchmark.py --seeds 1 --max-seq-len 4096
```

This runs a quick benchmark with reduced parameters to verify everything works.

## Results Interpretation

### Runtime Scaling

CRE demonstrates linear runtime scaling:
- L=512 → L=32768 (64x increase): Runtime grows ~67x (sub-linear due to parallelization overhead at short sequences)
- Transformer: Grows quadratically, OOM at L=32768
- Flash Attention: ~179x growth (still quadratic, but optimized)

### Memory Efficiency

At L=16384:
- CRE: 266.5 MB
- Transformer: 32,943.5 MB (124x more)
- Flash: 319.5 MB (1.2x more)

### Quality Metrics

Bracket matching task (pair identification):
- CRE: 0.900 ± 0.0015 AUC
- Transformer: 0.925 ± 0.0011 AUC
- Flash: 0.920 ± 0.0010 AUC

The 2.5-percentage-point difference represents a modest trade-off for significant computational advantages. Importantly, all three architectures achieve identical Effective Context Length (ECL-90), indicating that CRE's information propagation capacity matches Transformer's.

## Repository Structure

```
CRE/
├── cre.py                      # Core architecture implementation
├── scientific_benchmark.py     # Comprehensive evaluation framework
├── README.md                   # This file
├── LICENSE                     # Academic/Commercial dual license
├── CONTRIBUTING.md             # Contribution guidelines
├── COMMERCIAL_LICENSE.md       # Commercial licensing info
├── .gitignore                  # Git ignore file
├── results/
│   ├── scientific_benchmark_results.json    # Benchmark results
│   ├── scientific_benchmark_results.png     # Visualization
│   └── results_table.tex                    # LaTeX tables
└── paper/
    ├── CRE_paper.tex                 # Technical paper (LaTeX)
    ├── CRE_Paper_PlainText.md        # Technical paper (Markdown)
    └── references.bib                # References
```

## Implementation Notes

**Important**: The reference implementation uses pure PyTorch for clarity and reproducibility. This accurately demonstrates O(n) memory complexity but is not optimized for maximum throughput.

For production deployment, consider:
- Custom CUDA kernels for cumulative operations
- Parallel scan algorithms
- Hardware-specific optimizations (similar to Flash Attention's approach)

The current implementation prioritizes:
- Correctness and reproducibility
- Clear demonstration of O(n) scaling
- Accessibility for research

## Theoretical Properties

### Complexity Analysis

| Operation | Transformer | CRE |
|-----------|-------------|-----|
| Time | O(n²d) | O(nd²) |
| Memory | O(n²) | O(n) |
| Context Access | All-to-all | Cumulative |

### Position Modulation vs Temporal Decay

CRE uses **position modulation** rather than per-step temporal decay. The factor exp(-λt) applies to the cumulative sum at each position, serving to:
1. Normalize growing cumulative sums
2. Create learnable position-dependent weighting
3. Enable the model to optimize position sensitivity

This is distinct from architectures like Mamba or RWKV that use per-contribution decay. CRE's approach is simpler but still achieves competitive results, suggesting that the push-pull differential mechanism itself is the key innovation.

### Effective Context Length

ECL-90 (positions containing 90% of influence) is identical for CRE and Transformer at all tested lengths, indicating equivalent information propagation capacity.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{hammenfors2025cre,
  title={Complex Resonance Embedding: Linear-Complexity Sequence Processing 
         via Push-Pull Dynamics},
  author={Hammenfors, Sten Daniel},
  journal={TechRxiv preprint},
  year={2025},
  doi={10.36227/techrxiv.XXXXX}
}
```

## License

This project uses dual licensing:

- **Academic/Research Use**: Free for non-commercial research, education, and open-source projects
- **Commercial Use**: Requires a commercial license agreement

Patent pending (Norwegian Industrial Property Office, 2024-2025).

For commercial licensing inquiries: daniel.hammenfors@gmail.com

See [LICENSE](LICENSE), [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md) for details.

## Contact

- **Author**: Sten Daniel Hammenfors, MD, PhD
- **Email**: daniel.hammenfors@gmail.com
- **Location**: Bergen, Norway

## Acknowledgments

This work was conducted as independent research. The author thanks the open-source machine learning community for foundational tools and inspiration.

---

**Note**: This is an initial release presenting architectural properties and benchmark results. Future work will include language modeling evaluation and domain-specific applications.
