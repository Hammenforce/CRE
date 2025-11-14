# Contributing to CRE Architecture

Thank you for your interest in contributing to the CRE (Complex Resonance Embedding) project!

## How to Contribute

### Reporting Issues

- Use the GitHub issue tracker
- Include a clear description of the problem
- Provide minimal reproducible example when possible
- Include Python version, PyTorch version, and hardware details

### Submitting Changes

1. **Fork the repository**
   ```bash
   git clone https://github.com/Hammenforce/CRE.git
   cd cre-architecture
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add docstrings for new functions/classes
   - Include type hints where appropriate
   - Add tests for new functionality

4. **Test your changes**
   ```bash
   # Run basic tests
   python cre.py
   python examples.py
   
   # Run benchmarks (if applicable)
   python benchmark.py --quick
   ```

5. **Commit and push**
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Provide a clear description of changes
   - Reference any related issues
   - Explain the motivation for the changes

## Code Style

### Python Style Guide

- Follow PEP 8
- Use 4 spaces for indentation
- Maximum line length: 88 characters (Black formatter style)
- Use descriptive variable names

### Documentation

- Add docstrings to all public functions and classes
- Use Google-style docstrings:

```python
def function_name(arg1: int, arg2: str) -> bool:
    """
    Brief description of function.
    
    Longer description if needed.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
    
    Returns:
        Description of return value
    
    Example:
        >>> result = function_name(42, "test")
        >>> print(result)
        True
    """
    pass
```

### Type Hints

Use type hints for function arguments and returns:

```python
from typing import List, Tuple, Optional

def process_sequence(
    x: torch.Tensor,
    lengths: Optional[List[int]] = None
) -> Tuple[torch.Tensor, dict]:
    pass
```

## Testing

### Adding Tests

When adding new features, include tests:

```python
def test_new_feature():
    """Test the new feature."""
    # Setup
    model = CRELayer(d_model=64)
    x = torch.randn(2, 100, 64)
    
    # Test
    output, _ = model(x)
    
    # Assert
    assert output.shape == x.shape
    assert not torch.isnan(output).any()
```

### Running Tests

```bash
# Basic functionality
python cre.py

# Examples
python examples.py

# Benchmarks (quick)
python benchmark.py --quick
```

## Areas for Contribution

### High Priority

1. **Optimization**
   - Faster CUDA kernels
   - Memory optimization
   - Gradient checkpointing

2. **Documentation**
   - Tutorial notebooks
   - More examples
   - Video tutorials

3. **Testing**
   - Unit tests
   - Integration tests
   - Continuous integration setup

### Medium Priority

4. **Features**
   - Distributed training support
   - Model parallelism
   - Quantization support

5. **Benchmarking**
   - More baseline comparisons
   - Additional tasks
   - Efficiency metrics

6. **Utilities**
   - Pre-trained model zoo
   - Configuration files
   - Training scripts

### Nice to Have

7. **Research Extensions**
   - Adaptive frequency allocation
   - Sparse patterns
   - Multi-modal extensions

8. **Integrations**
   - HuggingFace integration
   - ONNX export
   - TensorFlow port

## Research Contributions

If you use CRE in your research and find improvements or extensions:

1. Open an issue describing your findings
2. Submit a PR with the implementation
3. Include benchmark results
4. Update documentation

We're particularly interested in:
- Novel applications of CRE
- Architectural improvements
- Training techniques
- Theoretical insights

## Communication

- **Questions**: Open a GitHub issue with the "question" label
- **Bugs**: Open a GitHub issue with the "bug" label
- **Features**: Open a GitHub issue with the "enhancement" label
- **Email**: daniel.hammenfors@gmail.com (for private matters)

## Code of Conduct

Be respectful and constructive in all interactions. We're here to advance research and build useful tools.

## License

By contributing, you agree that:
- Your contributions for **academic/research use** will be licensed under the free academic license
- **Commercial use** of contributions requires the same commercial licensing as the main project
- You retain copyright of your contributions

See the main [LICENSE](LICENSE) file for details on the dual licensing model.

---

Thank you for contributing to CRE! 🚀
