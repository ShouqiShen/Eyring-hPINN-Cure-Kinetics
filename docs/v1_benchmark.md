# v1 Benchmark

The v1 model is retained as a frozen Arrhenius-style benchmark.

## Purpose

v1 provides a stable reference point for measuring whether v2 improves:

- Ratio interpolation and extrapolation in Leave-One-Ratio-Out validation.
- Molecular-structure generalisation in Leave-One-Molecule-Out validation.
- Physical parameter trends across formulation and homologous series.
- Prediction uncertainty and residual correction behaviour.

## Physics Summary

v1 uses a Kamal-Sourour-style cure-rate backbone with Arrhenius-like kinetic parameters. The key extracted quantities are effective activation energy `Ea`, pre-exponential factor `lnA`, reaction orders, and vitrification-related terms.

## Recommended Use

Use v1 as a benchmark artifact. Keep code changes minimal so that comparisons against v2 remain meaningful.
