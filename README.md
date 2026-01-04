# Simulation of the Two-Body Gravitational Problem

This repository contains a Python implementation of the classical two-body gravitational problem with optional first-order relativistic corrections. The project emphasizes numerical integration, modular physics modeling, and reproducible simulations via JSON configuration files. A full theoretical and numerical discussion is provided in the accompanying LaTeX PDF.

---

## Features

- Newtonian and relativistic gravitational models  
- Symplectic Velocity-Verlet time integration  
- JSON-based configuration for reproducible runs  
- Three-dimensional orbital trajectory visualization  
- Orbital element extraction from numerical data  

---

## Repository Structure

Simulation-of-the-Two-Body-Gravitational-Problem/
│

├── main.py                       # Primary simulation entry point

├── integrator.py                 # Velocity–Verlet time integrator

├── orbital_elements.py           # Orbital element calculations

│

├── analytic_kepler_solver.py     # Analytical Kepler orbit solutions

├── numerical_vs_analytical.py    # Numerical vs analytical orbit comparison

├── newtonian_vs_relativistic.py  # Newtonian vs relativistic dynamics comparison

│

├── configs/                      # Example simulation input files

│   ├── elliptic_ex.json

│   ├── hyperbolic_ex.json

│   └── parabolic_ex.json

│

├── docs/                         # Documentation and write-up

│   └── Write_Up.pdf

│

├── .gitignore

└── README.md



Generated plots and animations are intentionally excluded from version control.

---

## Usage

Run a simulation by supplying a JSON configuration file:

```bash
python main.py input_example.json
```
All physical parameters, initial conditions, and time-stepping settings are defined in the input JSON files.
