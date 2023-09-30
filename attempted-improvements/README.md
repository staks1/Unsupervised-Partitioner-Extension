# Hierarchical Navigable Small World (HNSW) in PyTorch

This repository contains the outline of a basic implementation of the Hierarchical Navigable Small World (HNSW) algorithm in PyTorch. HNSW is an efficient data structure for approximate nearest neighbor search in high-dimensional spaces.

## Table of Contents

- [Background](#background)
- [Implementation](#implementation)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Usage](#usage)
- [References](#references)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Background

One of the key enhancements that was intended for this project was the integration of the Hierarchical Navigable Small Worlds (HNSW) algorithm. Finding nearest neighbors in high-dimensional spaces is a common problem in machine learning, data mining, and search systems. HNSW could potentially provide an efficient solution for approximate nearest neighbor search. It constructs a hierarchical graph in which each level is a sub-sample of the lower level, which allows for faster navigation through the space.

However, due to the algorithm's complexity and the time constraints faced during the development process, it was not feasible to fully implement the HNSW algorithm. Nevertheless, the class developed as part of this attempt has been included in the repository for reference and future development.
 
## Implementation

This project provides a skeleton for implementing HNSW in PyTorch. The code includes the foundation for methods to insert data points and perform a navigable search, but the core logic has not as yet been implemented.

## Getting Started

### Prerequisites

- Python 3.7
- PyTorch

You can install PyTorch using pip:

```shell
pip install torch
```

### Usage

First, clone the repository to your local machine:

```shell
git clone https://github.com/DSIT-DB-Course/phase-3-implementation-darmanis_kotsis/main/hnsw-pytorch.git
cd hnsw-pytorch
```

Use the HNSW class in your Python script:

```python
from model import HNSW

# Initialize HNSW graph
graph = HNSW(max_elements=1000, dim=128)

# Insert data points and perform search (implement methods first)
```

## References

- [HNSW Original Paper](https://arxiv.org/abs/1603.09320)

## Acknowledgments

- The authors of the HNSW paper for developing the algorithm.
