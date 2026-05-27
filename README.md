# Educational Code

This repository is a collection of small scripts, examples, and benchmarks that
help explore random technical concepts. Each subdirectory is meant to be
self-contained and focused on a specific idea, with its own README describing
the motivation, setup, and results.

## Projects

- [Riccati LQR Example](./riccati/README.md): a compact Python example for
  solving a finite-horizon and infinite-horizon LQR problem with Riccati recursions.
- [Robust Quadratic Formula](./robust-quadratic-formula/README.md): a small C
  example showing how the citardauq formula avoids catastrophic cancellation in
  a step-length computation near a second-order cone boundary.
- [Batch Sparse Linear System Solver Benchmark](./batch_linsys/README.md): a
  Python benchmark comparing serial CPU sparse solves and batched GPU solves for
  KKT-style linear systems from finite-horizon LQR problems.
