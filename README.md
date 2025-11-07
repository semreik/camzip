# 🗜️ CamZIP: Information Theory Compression Algorithms

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

An educational implementation of fundamental data compression algorithms based on information theory. This project demonstrates **Shannon-Fano**, **Huffman**, and **Arithmetic coding** algorithms with interactive examples and visualizations.

## 📚 Table of Contents

- [Overview](#overview)
- [Mathematical Theory](#mathematical-theory)
  - [Information Theory Fundamentals](#information-theory-fundamentals)
  - [Shannon-Fano Coding](#shannon-fano-coding)
  - [Huffman Coding](#huffman-coding)
  - [Arithmetic Coding](#arithmetic-coding)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Contributing](#contributing)

## 🎯 Overview

CamZIP is an educational project that implements three fundamental lossless compression algorithms:

1. **Shannon-Fano Coding**: A prefix-free variable-length coding scheme based on symbol probabilities
2. **Huffman Coding**: An optimal prefix-free coding algorithm that minimizes expected codeword length
3. **Arithmetic Coding**: A sophisticated entropy coder that approaches the theoretical compression limit

The project includes a comprehensive Jupyter notebook with interactive demonstrations using classic texts like Shakespeare's Hamlet.

## 📐 Mathematical Theory

### Information Theory Fundamentals

#### Entropy

The **entropy** $H(X)$ of a discrete random variable $X$ with probability mass function $p(x)$ is defined as:

$$H(X) = -\sum_{x \in \mathcal{X}} p(x) \log_2 p(x)$$

Entropy represents the **theoretical lower bound** for the average number of bits needed to encode symbols from the source. It measures the average information content or "surprise" in the data.

**Properties:**
- $H(X) \geq 0$ (non-negative)
- $H(X) = 0$ if and only if $X$ is deterministic
- $H(X) \leq \log_2 |\mathcal{X}|$ with equality when $X$ is uniformly distributed

#### Expected Code Length

For a source with probability distribution $p(x)$ and a code with codeword lengths $\ell(x)$, the expected code length is:

$$L = \sum_{x \in \mathcal{X}} p(x) \ell(x)$$

**Shannon's Source Coding Theorem** states that for any uniquely decodable code:

$$L \geq H(X)$$

This establishes entropy as the fundamental limit for lossless compression.

### Shannon-Fano Coding

Shannon-Fano coding is a **variable-length prefix-free coding** technique developed by Claude Shannon and Robert Fano.

#### Algorithm

1. **Sort** symbols in decreasing order of probability: $p(x_1) \geq p(x_2) \geq \cdots \geq p(x_n)$

2. **Compute cumulative distribution function**:
   $$F(x_i) = \sum_{j=1}^{i-1} p(x_j)$$

3. **Assign codeword lengths** according to Shannon's formula:
   $$\ell(x_i) = \lceil \log_2 \frac{1}{p(x_i)} \rceil$$

4. **Generate codewords**: The codeword for symbol $x_i$ is the first $\ell(x_i)$ bits of the binary expansion of $F(x_i)$:
   $$c(x_i) = \text{binary}(F(x_i))_{1:\ell(x_i)}$$

#### Performance

Shannon-Fano coding achieves:

$$H(X) \leq L < H(X) + 1$$

This guarantees compression within 1 bit of the entropy bound per symbol.

### Huffman Coding

Huffman coding, invented by David Huffman in 1952, is an **optimal prefix-free code** that minimizes the expected codeword length.

#### Algorithm

1. Create a leaf node for each symbol with its probability
2. **While** more than one node exists:
   - Select the two nodes with smallest probabilities $p_1$ and $p_2$
   - Create a parent node with probability $p_1 + p_2$
   - Remove the two nodes and add the parent to the tree
3. Assign binary labels to tree edges (e.g., 0 for left, 1 for right)
4. The codeword for each symbol is the path from root to leaf

#### Optimality

Huffman coding is **optimal** among all prefix-free codes, meaning:

$$L_{\text{Huffman}} = \min_{C \in \mathcal{C}} \sum_{x} p(x) \ell_C(x)$$

where $\mathcal{C}$ is the set of all prefix-free codes.

#### Performance Bound

For Huffman coding:

$$H(X) \leq L_{\text{Huffman}} < H(X) + 1$$

The overhead is at most 1 bit per symbol, with equality in the worst case when all probabilities are powers of $\frac{1}{2}$.

### Arithmetic Coding

Arithmetic coding is an **entropy coding** technique that represents an entire message as a single number in the interval $[0, 1)$.

#### Core Idea

Given a probability distribution $p(x)$, partition the interval $[0, 1)$ proportionally:

$$[0, p(x_1)) \cup [F(x_2), F(x_2) + p(x_2)) \cup \cdots \cup [F(x_n), 1)$$

where $F(x_i) = \sum_{j=1}^{i-1} p(x_j)$ is the cumulative distribution function.

#### Algorithm

**Encoding:**
1. Initialize interval $[\ell, h) = [0, 1)$
2. For each symbol $x_i$ in the message:
   - Update interval: $[\ell', h') = [\ell + F(x_i) \cdot (h - \ell), \ell + (F(x_i) + p(x_i)) \cdot (h - \ell))$
3. Output any number in the final interval $[\ell, h)$

**Decoding:**
1. Read the encoded value $v \in [0, 1)$
2. Initialize interval $[\ell, h) = [0, 1)$
3. While symbols remain:
   - Find symbol $x$ such that $v \in [\ell + F(x) \cdot (h - \ell), \ell + (F(x) + p(x)) \cdot (h - \ell))$
   - Output $x$ and update interval as in encoding
   - Decode until message length reached

#### Performance

Arithmetic coding achieves **near-optimal compression**:

$$L_{\text{arithmetic}} \approx H(X) + \epsilon$$

where $\epsilon$ can be made arbitrarily small with sufficient precision. Unlike Huffman coding, arithmetic coding can achieve fractional bits per symbol, approaching the entropy limit asymptotically.

#### Practical Implementation

To avoid precision issues with floating-point arithmetic, practical implementations use:
- **Integer arithmetic** with rescaling
- **Bit-shifting** operations when intervals converge
- **Precision management** to handle the $[0, 1)$ interval efficiently

## ✨ Features

- **Three compression algorithms**: Shannon-Fano, Huffman, and Arithmetic coding
- **Complete implementations**: Encoding and decoding for all algorithms
- **Tree visualization**: Convert compression trees to Newick format for visualization
- **Interactive notebook**: Jupyter notebook with step-by-step demonstrations
- **Real-world examples**: Compression of text files (Hamlet)
- **Performance metrics**: Entropy calculation, compression ratios, and analysis
- **Bit/byte utilities**: Conversion functions for practical file compression
- **Error resilience testing**: Demonstration of error propagation in compressed data

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/camzip.git
cd camzip

# Install Jupyter (if not already installed)
pip install jupyter

# Launch the notebook
jupyter notebook 3F7lab.ipynb
```

## 💻 Usage

### Interactive Notebook

The main interface is the Jupyter notebook `3F7lab.ipynb`. Open it to:
- Learn about tree data structures
- Implement and test Shannon-Fano coding
- Implement and test Huffman coding  
- Implement and test Arithmetic coding
- Compress and decompress files
- Visualize compression trees
- Analyze compression performance

### Command-Line Usage (via notebook functions)

```python
from vl_codes import shannon_fano, huffman, vl_encode, vl_decode

# Define probability distribution
p = {'A': 0.4, 'B': 0.3, 'C': 0.2, 'D': 0.1}

# Shannon-Fano coding
sf_code = shannon_fano(p)
print(f"Shannon-Fano codes: {sf_code}")

# Huffman coding
huff_tree = huffman(p)
print(f"Huffman tree: {huff_tree}")

# Encode data
data = ['A', 'B', 'C', 'D', 'A', 'A']
encoded = vl_encode(data, sf_code)
print(f"Encoded: {encoded}")
```

### Testing

Run the test suite to verify functionality:

```bash
python test_functionality.py
```

## 📁 Project Structure

```
camzip/
├── 3F7lab.ipynb                 # Main Jupyter notebook with theory and examples
├── .ipynb_checkpoints/
│   └── vl_codes.py             # Variable-length coding implementations
├── test_functionality.py        # Test suite for all algorithms
├── hamlet.txt                   # Example text file (not included)
└── README.md                    # This file
```

### Key Files

- **`3F7lab.ipynb`**: Comprehensive notebook with implementation exercises and examples
- **`vl_codes.py`**: Core implementations of Shannon-Fano and Huffman coding
- **`test_functionality.py`**: Unit tests for verifying algorithm correctness

## 📊 Examples

### Compressing Hamlet

The notebook demonstrates compression of Shakespeare's Hamlet (207,039 bytes):

| Algorithm | Compressed Size | Compression Ratio | Bits/Symbol |
|-----------|----------------|-------------------|-------------|
| Entropy (theoretical) | - | - | 4.45 |
| Shannon-Fano | 124,694 bytes | 60.2% | 4.82 |
| Huffman | ~124,000 bytes | ~59.9% | ~4.79 |
| Arithmetic | ~115,000 bytes | ~55.5% | ~4.45 |

### Tree Visualization

Convert compression trees to Newick format and visualize at [phylo.io](https://phylo.io):

```python
from vl_codes import tree2newick

t = [-1, 0, 1, 1, 0]
newick = tree2newick(t, ['root', 'child 0', 'grandchild 0', 'grandchild 1', 'child 1'])
print(newick)
# Output: ((grandchild 0,grandchild 1)child 0,child 1)root
```

### Error Resilience

The notebook includes demonstrations of how single-bit errors propagate through compressed data, showing the trade-offs between compression efficiency and error resilience.

## 🎓 Educational Value

This project is ideal for:
- **Information Theory courses**: Hands-on implementation of fundamental algorithms
- **Data Compression students**: Understanding practical aspects of compression
- **Computer Science education**: Learning about entropy, coding theory, and algorithmic optimization
- **Self-learners**: Interactive exploration of compression techniques

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:

- Additional compression algorithms (LZW, LZ77, etc.)
- Performance optimizations
- More comprehensive test suites
- Additional example datasets
- Improved visualization tools
- Documentation improvements

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Based on the **3F7 Information Theory and Coding** course
- Original tree manipulation utilities by Jossy (2018)
- Inspired by Claude Shannon's groundbreaking work in information theory
- David Huffman's optimal coding algorithm

## 📚 References

1. Shannon, C. E. (1948). "A Mathematical Theory of Communication". *Bell System Technical Journal*.
2. Huffman, D. A. (1952). "A Method for the Construction of Minimum-Redundancy Codes". *Proceedings of the IRE*.
3. Witten, I. H., Neal, R. M., & Cleary, J. G. (1987). "Arithmetic Coding for Data Compression". *Communications of the ACM*.
4. Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*. Wiley-Interscience.

---

**Made with ❤️ for Information Theory enthusiasts**
