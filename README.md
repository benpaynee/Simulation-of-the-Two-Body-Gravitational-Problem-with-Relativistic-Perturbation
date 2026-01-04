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

├── main.py # Simulation entry point

├── integrator.py # Velocity-Verlet integrator

├── orbital_elements.py # Orbital element calculations

├── gravity_models.py # Gravitational acceleration models

│

├── configs/ # Example simulation inputs

│ ├── example_newtonian.json

│ └── example_relativistic.json

│

├── docs/ # LaTeX write-up and curated figures

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
