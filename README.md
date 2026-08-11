# Rydberg Atom Receiver Simulations & Quantum Machine Learning (QML)

A quantum simulation and quantum machine learning framework modeling **Rydberg atomic receivers** for RF/microwave electric and magnetic field sensing, evaluating fundamental quantum metrology bounds (**Quantum Fisher Information / Quantum Cramér-Rao Bound**), and performing **continuous parameter estimation using Variational Quantum Circuits (PennyLane)**.

---

## 📌 Project Overview

Rydberg atom electrometry leverages the extreme electric dipole moments and polarizability of high-$n$ atomic Rydberg states to perform highly sensitive, SI-traceable radio-frequency (RF) and microwave electric field detection via **Electromagnetically Induced Transparency (EIT)** and **Autler-Townes (AT) splitting**.

This repository provides:
1. **Open Quantum Systems Dynamics (QuTiP):** Solving the Lindblad master equation for multi-level cascade atomic systems (4-level and 6-level Zeeman-resolved manifolds).
2. **Quantum Metrology & Fundamental Limits (QFI / QCRB):** Computing the Quantum Fisher Information (QFI) for mixed density matrices via the Šafránek metric to determine ultimate angular and amplitude estimation precision ($\Delta \theta \ge 1 / \sqrt{\nu \mathcal{F}_Q}$).
3. **Qudit-to-Qubit Embedding & Gate Evolution:** Mapping arbitrary-dimensional open-system density matrices (4x4 $\to$ 2 qubits, 6x6 padded $\to$ 8x8 $\to$ 3 qubits) to PennyLane-compatible density matrices with physical validity constraints (Hermiticity, unit trace, positive semidefiniteness).
4. **Hybrid Quantum-Classical Regression (QML):** Parameterized Quantum Circuit (PQC / VQC on `default.mixed`) combined with a classical linear readout head to estimate continuous parameters (e.g., incoming RF angle $\theta_{\text{RF}}$) directly from atomic steady states under noisy detuning conditions.

---

## 🔬 Theoretical Foundations

### 1. Atomic Master Equation
The steady-state density matrix $\rho_{\text{ss}}$ of the driven Rydberg system is determined by the Lindblad master equation:
$$\frac{d\rho}{dt} = -\frac{i}{\hbar}[H, \rho] + \sum_k \mathcal{D}[L_k]\rho = 0$$
where $H$ includes probe ($\Omega_p$), coupling laser ($\Omega_c$), and RF field ($\Omega_{\text{RF}}$) drives under the Rotating Wave Approximation (RWA), and $L_k$ are collapse operators for radiative decay and dephasing.

### 2. Quantum Fisher Information (QFI) for Mixed States
For parameter $\theta_{\text{RF}}$, the Quantum Fisher Information $\mathcal{F}_Q$ for mixed state $\rho$ is computed using the vectorized symmetric logarithmic derivative representation:
$$\mathcal{F}_Q = 2 \operatorname{Re}\left[ \operatorname{vec}\left(\frac{\partial \rho}{\partial \theta}\right)^\dagger \left( \rho^* \otimes I + I \otimes \rho \right)^{-1} \operatorname{vec}\left(\frac{\partial \rho}{\partial \theta}\right) \right]$$
The corresponding Quantum Cramér-Rao Bound (QCRB) sets the precision limit:
$$\delta \theta \ge \frac{1}{\sqrt{\nu \mathcal{F}_Q}}$$

### 3. QML Angle Regression Architecture
```
  [QuTiP Steady-State ρ_ss] 
             │
             ▼
  [Strict / Padded State Converter]  (Hermiticity, Tr(ρ)=1, Dim = 2^n)
             │
             ▼
  [PennyLane QubitDensityMatrix] (default.mixed simulator)
             │
             ▼
  [Variational Circuit: RY / RZ + CNOT Entangler]
             │
             ▼
  [Expectation Values <Z_0>, <Z_1>, <X_0>]
             │
             ▼
  [Classical Linear Readout Head (w·f + b)] ──► Predicted Angle θ_hat
```

---

## 📂 Repository Structure

```bash
├── qcrb_simulation_and_dataset.py      # 6-level Zeeman-resolved model, QFI/QCRB sweeps vs E0 & B, dataset generator
├── qml_regression_demo.py              # End-to-end QML regression demo (QuTiP + PennyLane + Adam optimizer)
├── qutip_pennylane_conversion.py       # Core utility converting QuTiP Qobjs to PennyLane qubit density matrices
├── conversion_smoke_tests.py           # Unit tests validating mathematical & physical invariants across sweeps
├── pennylane_gate_smoke_tests.py       # Mixed-state gate stack verification for 2-qubit (4x4) and 3-qubit (8x8) states
├── apply_gate_on_padded_density_example.py # Tutorial: 6x6 -> 8x8 padded density matrix evolution in PennyLane
├── easy_verify_padded_gate_example.py  # Verification comparing PennyLane gate outputs with analytic U ρ U†
├── demo_QCRB_vs_E_and_B.ipynb          # Interactive notebook analyzing QCRB resolution vs E0 and B fields
├── Starting Point - Solving the Master Equation/
│   ├── rho21_vs_deltac_collapse_operators.py # 4-level Lindblad solver with collapse operators vs analytical formula
│   └── rho21_vs_deltac_superoperator.py      # 4-level master equation solved via superoperator/Liouvillian matrix
├── Next Step - Replicate Chen et al (August25) paper results/
│   └── figure_3_simulation.py         # Zeeman-resolved EIT spectrum replication (Chen et al. paper)
├── requirements.txt                    # Python dependencies
└── .gitignore                          # Standard git ignore configuration
```

---

## 🚀 Getting Started

### Installation
Clone the repository and install required packages:
```bash
git clone https://github.com/your-username/Rydberg_Receiver_Simulations.git
cd Rydberg_Receiver_Simulations
pip install -r requirements.txt
```

### Running Simulations & Demos

1. **Run the QML Regression Pipeline:**
   ```bash
   python qml_regression_demo.py --samples 150 --steps 60 --lr 0.05
   ```

2. **Compute QFI / QCRB Sweeps:**
   ```bash
   python qcrb_simulation_and_dataset.py
   ```

3. **Run Validation & Smoke Tests:**
   ```bash
   python conversion_smoke_tests.py
   python pennylane_gate_smoke_tests.py
   ```

4. **Explore Interactive Notebook:**
   ```bash
   jupyter notebook demo_QCRB_vs_E_and_B.ipynb
   ```

---

## 🛠 Tech Stack
- **Quantum Simulation:** QuTiP (Quantum Toolbox in Python)
- **Quantum Machine Learning:** PennyLane, Autograd
- **Scientific Computing:** NumPy, SciPy, SymPy (Wigner 3j symbols)
- **Visualization:** Matplotlib