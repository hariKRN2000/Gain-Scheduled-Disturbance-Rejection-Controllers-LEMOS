# Gain-Scheduled Optogenetic Feedback for Disturbance Rejection in Bacterial Batch Cultures

This repository contains the simulation code and experimental data accompanying the paper by Hari Namboothiri and Chelsea Hu, submitted to CDC 2026.

---

## Repository Structure

```
.
├── model_and_controller_equations/
│   ├── TCS_model_growth_equations.py   # ODE model of the TCS optogenetic circuit coupled to logistic cell growth
│   ├── run_constant.py                 # Runs the model under constant light input (no feedback)
│   └── run_all_controllers.py          # Implements all controllers (P, PI, PD, PID, PID-GS, FF, PID-GS-FF) and the closed-loop simulation loop
│
├── frequency response analysis/
│   └── Sim_TF_Bode_C_as_Disturbance.ipynb   # Frequency response analysis treating cell growth as a disturbance input
│
├── experiment_data/
│   └── P-FL_OD_run_data_*.csv          # Raw OD timeseries from optogenetic expression experiments
│
├── parameters/
│   └── TCS_params_guess_070225.csv     # Fitted model parameters (kinetic rates and initial conditions)
│
├── sweep_results/
│   └── sweep_results_7_by_7.csv        # Results from a 7×7 gain/tau parameter sweep
│
├── figures/                            # Pre-generated SVG figures used in the paper
│
├── perturbation_simulation_fixed_gain_PID.ipynb    # Simulates growth perturbations under fixed-gain PID
├── perturbation_simulation_all_controllers.ipynb   # Compares all controllers under the same perturbation conditions
├── plot_performance_metrics.ipynb                  # Generates performance metric heatmaps (rise time, settling time, ITAE)
└── signal_analysis.py                              # Helper functions for computing control performance metrics
```

---

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install numpy scipy pandas lmfit matplotlib jupyter
```

Then launch Jupyter and run any of the `.ipynb` notebooks.
