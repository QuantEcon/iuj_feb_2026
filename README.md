# Computational Methods for Simulation

## An Analysis of the Schelling Model

[![Build & Publish](https://github.com/QuantEcon/iuj_feb_2026/actions/workflows/publish.yml/badge.svg)](https://github.com/QuantEcon/iuj_feb_2026/actions/workflows/publish.yml)

**Prepared for the International University of Japan**

**Author: [John Stachurski](https://johnstachurski.net/)**

This workshop explores simulation for economic analysis.

It demonstrates how to accelerate simulations using modern Python tools.

This workshop focuses on Thomas Schelling's segregation model as our running example.

The key ideas can be applied in other settings.

## Overview

The Schelling segregation model shows how mild individual preferences can lead to extreme aggregate outcomes. We study:

1. **The basic model** — Understanding the dynamics of racial segregation using Python classes
2. **NumPy implementation** — Rewriting the model with arrays and functions for clarity and speed
3. **JAX implementation** — Translating the model to JAX syntax and concepts
4. **Further parallelization** — How can we exploit modern parallel hardware (e.g., GPUs)

## Requirements

- Python 3.13
- Anaconda
- GPU support (optional, for JAX acceleration)

See [environment.yml](environment.yml) for the full list of dependencies.

## License

This work is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).
