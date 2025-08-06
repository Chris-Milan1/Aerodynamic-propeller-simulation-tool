# Aerodynamic-propeller-simulation-tool
This is a Python tool. Designed using the BEM theory to compute key performance metrics for propeller aerodynamic behaviour
High-Efficiency Propeller Design for Human-Powered Aquatic Vehicle. It covered the full engineering workflow—from aerodynamic modeling and simulation to physical prototyping and lab testing. 

#  High-Efficiency Propeller Simulation for Human-Powered Aquatic Vehicle
This project presents a high-fidelity simulation of a **custom-designed propeller** intended for a human-powered aquatic vehicle. It models the **aerodynamic performance**, **mechanical response**, and **time-evolving system dynamics** using a combination of physics-based models and numerical methods. The codebase was developed entirely in Python and reflects a complete applied engineering workflow — from theoretical modeling to simulation and CAD-ready output.

The core objective was to evaluate and optimize propeller geometry (e.g., blade twist, chord, radius) to maximize **efficiency** and **thrust** under human power constraints, while modeling how the vehicle accelerates over time.


Developed as part of a technical engineering project, this code reflects skills relevant to **quantitative research**, including:
- Mathematical modeling
- Python-based simulation
- Data analysis
- Differential system modeling

---

## Features

###  Blade Element Momentum (BEM) Theory
- Implements a discretized version of BEM to resolve local aerodynamic forces along the blade span.
- Models inflow angle, induction factor, and pressure distribution per segment.

### Differential System Modeling
- Models the time-dependent motion of the vehicle using first-order ODEs:  
  $\displaystyle m \frac{dv}{dt} = T(v) - D(v)$
- Solves for boat velocity $v(t)$ over time using a dynamic thrust-drag balance.
- Includes nonlinear damping and drag terms, dependent on velocity and geometry.

### Numerical Methods
- Uses Brent’s method for root-finding ($f(x) = 0$) where analytical solutions are not available.
- Applies adaptive time stepping for dynamic simulations to ensure numerical stability.
- Polynomial regression for fitting empirical airfoil data:  
  $C_L(\alpha)$ and $C_D(\alpha)$ as functions of angle of attack $\alpha$.

### High-Performance Computation
- Vectorized operations using **NumPy** for fast evaluation of force and performance equations across blade elements.
- Supports batched simulation runs across multiple parameter sets (e.g., RPM, chord, twist).
- Structured for scalability and future parallelization (e.g., with `multiprocessing` or `numba`).

### Parametric Sweep & Optimization
- Systematically varies key parameters:
  - Blade radius $R$, twist angle $\theta(r)$, and chord $c(r)$
  - Rotational speed $\omega$, mass $m$, and power input $P$
- Outputs performance metrics per run:
  - Thrust $T$, Torque $\tau$, Velocity $v(t)$, Efficiency $\eta$

### Output Files
- `Main_results.csv`: Summary of simulation outcomes for each run.
- `Scale factor *.csv`: Time-resolved simulation data for each condition.
- `section_*.txt`: Airfoil blade geometry for CAD export.

### SolidWorks Geometry Export (Optional)
- Automatically exports 3D coordinates of blade sections along the span.
- Can be directly imported into SolidWorks or other CAD tools for prototyping.
---

##  How to Run

### 1. Prepare Airfoil Data
Provide a CSV file with airfoil data, like:

```csv
Alpha,Cl,Cd
-5,0.1,0.02
0,0.3,0.01
5,0.6,0.015
...
```

### 2. Run the Simulation 
Use the terminal or command prompt. Once prompted, decide whether or not you would like to receive SolidWorks coordinates 

---
## Author
Developed as part of an engineering research project focused on aerodynamic optimization for human-powered aquatic propulsion. This repository reflects interests in simulation, numerical modeling, and applied quantitative analysis. 

---

## **⚠️ Disclaimer (Important)**

As with any simulation, results may vary depending on the geometry and parameters used. There is no guarantee that a simulated performance will exactly reflect real-world behavior. This project is purely for academic and theoretical use; further research must be conducted before any further use (Practical or theoretical). 

The author assumes no responsibility for any misuse, misinterpretation, or unintended consequences arising from the use of this code.
