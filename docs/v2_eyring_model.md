# v2 Eyring Model

The v2 model is the main hPINN implementation in this repository.

## Purpose

v2 aims to improve the physical interpretability and automation of cure-kinetics prediction by replacing the v1 Arrhenius-style backbone with an Eyring transition-state-theory formulation.

## Physics Summary

The model learns physically bounded activation enthalpy and activation entropy terms, combines them with reaction-order terms, and applies diffusion-aware gating for vitrification effects. Residual heads model remaining deviations from the physics backbone, and MC dropout is used to estimate predictive uncertainty.

## Recommended Use

Use v2 when the material system and processing conditions are inside the documented validity domain. For extrapolation, report uncertainty and compare against the v1 benchmark.
