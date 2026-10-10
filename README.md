# minimicro-tensor-compiler
A custom tensor graph compiler and runtime engine demonstrating IR transformations, operator fusion passes, and memory bandwidth optimization.

# Micro-Tensor Compiler

A high-performance deep learning graph compiler and runtime engine featuring intermediate representation (IR), constant folding, and automated **operator fusion** (similar to XLA and TVM architectures).

## Architecture

[ONNX / PyTorch Graph] ──> [Canonicalization] ──> [Operator Fusion Pass] ──> [Optimized IR Execution]


## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run test suite
make test

# Run memory bandwidth benchmarks
make bench