# CRE: Complex Resonance Embedding

[![License: Academic Free / Commercial](https://img.shields.io/badge/License-Academic%20Free%20%2F%20Commercial-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

A novel sequence processing architecture achieving **O(n) complexity** through push-pull cumulative dynamics and multi-frequency resonance.

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
CRE:               O(n)  - cumulative push-pull with exponential decay
```

### Core Mechanism

```python
# Push-Pull Dynamics
P_t = α * P_{t-1} + g_t * v_t      # Push state (accumulation)
Q_t = α * Q_{t-1} + (1-g_t) * v_t  # Pull state (subtraction)
h_t = P_t - Q_t                     # Resonance output

# Where:
# α = exp(-λ) is learnable decay
# g_t = sigmoid(W_g * x_t) controls information flow
# v_t = W_v * x_t is the value projection
```

### Key Innovations

1. **Push-Pull Dynamics**: Differential accumulation enables selective information retention
2. **Multi-Frequency Resonance**: Multiple decay rates capture both short and long-term patterns
3. **Gated Information Flow**: Learnable gates control what information enters each state
4. **Linear Complexity**: All operations are O(n) in sequence length

## Installation

```bash
git clone https://github.com/Hammenforce/CRE.git
cd cre-architecture
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
- L=512 → L=32768 (64x increase): Runtime grows 34x (sub-linear due to parallelization overhead at short sequences)
- Transformer: Grows quadratically, OOM at L=32768
- Flash Attention: 125x growth (still quadratic, but optimized)

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
cre-architecture/
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
    ├── CRE_paper.tex                 # Technical paper
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

### Effective Context Length

ECL-90 (positions containing 90% of influence) is identical for CRE and Transformer at all tested lengths, indicating equivalent information propagation capacity.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{hammenfors2025cre,
  title={Complex Resonance Embedding: Linear-Complexity Sequence Processing 
         via Push-Pull Dynamics},
  author={Hammenfors, Sten Daniel},
  journal={arXiv preprint arXiv:2501.XXXXX},
  year={2025}
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
