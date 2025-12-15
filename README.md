# ease-optimization
# Optimizing EASE via Linear System Solvers

This repository investigates computationally efficient modifications of the
EASE (Embarrassingly Shallow Autoencoder) algorithm by replacing explicit matrix
inversion with direct and iterative solvers for systems of linear algebraic equations (SLAE).

The project focuses on numerical stability, scalability, incremental updates,
and differentiable integration into modern ML pipelines.

---

## Repository Structure

The repository is organized to separate theoretical analysis, algorithmic implementations,
incremental methods, and experimental evaluation.

- `src/ease/`  
  Core EASE implementations based on solving linear systems instead of explicit matrix
  inversion. Includes baseline inversion, direct solvers (Cholesky), and iterative methods
  (CG, PCG, LSQR, AMG).

- `src/torch_ease/`  
  Differentiable EASE implementation in PyTorch. Solves linear systems inside the forward
  pass and supports gradient backpropagation via implicit differentiation.

- `src/incremental/`  
  Incremental and online update methods for EASE, including Woodbury-based and low-rank
  updates, enabling weight updates without full retraining.

- `experiments/`  
  Benchmarking and evaluation code for comparing solvers in terms of runtime, memory
  usage, numerical stability, and recommendation quality (Recall@K, NDCG@K).

- `docs/`  
  Theoretical background, numerical analysis, and discussion of related work.

- `notebooks/`  
  Exploratory analysis and visualizations (not used for final experiments).

- `tests/`  
  Unit tests for solver correctness, numerical equivalence, and autograd validation.

