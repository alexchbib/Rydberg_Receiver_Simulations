# Rydberg Atom Receiver Simulations & Quantum Machine Learning (QML)

A quantum simulation and quantum machine learning (QML) framework modeling **Rydberg atomic receivers** for RF/microwave electric and magnetic field sensing, evaluating fundamental quantum metrology limits (**Quantum Fisher Information / Quantum Cramér-Rao Bound**), and performing **continuous parameter estimation using Variational Quantum Circuits (PennyLane)**.

---

## 🏛️ System Architecture & Division of Modules

This repository is organized to maintain a clear boundary between **Domain Physics Specifications** (atomic Hamiltonians, transition matrices, physical decay channels) and the **Quantum Software & QML Engineering Pipeline** (numerical solvers, state embedding bridges, variational circuits, and automated testing).

```
Rydberg_Receiver_Simulations/
│
├── 🧠 QUANTUM SOFTWARE & QML ENGINE (Core Implementations):
│   ├── qutip_pennylane_conversion.py   # State Conversion Bridge: Validates & embeds density matrices into PennyLane
│   ├── qml_regression_demo.py          # End-to-end Variational Quantum Circuit (VQC) regression pipeline
│   ├── qcrb_simulation_and_dataset.py  # Numerical steady-state solver, QFI metrology bounds & dataset generation
│   └── test_suite.py                   # Unified verification suite (mathematical invariants & gradient checks)
│
├── ⚛️ DOMAIN PHYSICS SPECIFICATIONS (Theoretical Formulations & Parameters):
│   └── physics_models/
│       ├── four_level_model.py         # 4-level cascade Hamiltonian (RWA) & Lindblad decay operators
│       ├── six_level_zeeman_model.py   # 6-level Zeeman manifold, dipole constants, and Wigner-3j / CG matrices
│       └── paper_benchmarks.py         # Analytical continued-fraction formulas (Chen et al. literature replication)
│
├── 📊 VISUALIZATION & CONFIGURATION:
│   ├── demo_QCRB_vs_E_and_B.ipynb      # Interactive notebook for parameter sweeps and QCRB plots
│   ├── requirements.txt                # Python dependencies
│   └── .gitignore                      # Git configuration
```

---

## 🔬 Physics & Engineering Pipeline

### 1. Open Quantum Systems Simulation
The steady-state density matrix $\rho_{\text{ss}}$ is obtained by solving the Lindblad master equation:
$$\frac{d\rho}{dt} = -\frac{i}{\hbar}[H, \rho] + \sum_k \left( L_k \rho L_k^\dagger - \frac{1}{2} \{ L_k^\dagger L_k, \rho \} \right) = 0$$

### 2. Quantum Fisher Information (QFI) & Precision Bounds
For incoming RF polarization angle $\theta_{\text{RF}}$, the Quantum Fisher Information $\mathcal{F}_Q$ is evaluated for mixed states via the vectorized symmetric logarithmic derivative representation:
$$\mathcal{F}_Q = 2 \operatorname{Re}\left[ \operatorname{vec}\left(\frac{\partial \rho}{\partial \theta}\right)^\dagger \left( \rho^* \otimes I + I \otimes \rho \right)^{-1} \operatorname{vec}\left(\frac{\partial \rho}{\partial \theta}\right) \right]$$
Setting the Quantum Cramér-Rao Bound (QCRB):
$$\delta \theta \ge \frac{1}{\sqrt{\nu \mathcal{F}_Q}}$$

### 3. State Conversion & Qudit-to-Qubit Embedding Bridge
PennyLane requires $2^n$ qubit Hilbert space representations. The conversion engine (`qutip_pennylane_conversion.py`):
- Enforces strict physical invariants: $\rho = \rho^\dagger$ (Hermiticity), $\operatorname{Tr}(\rho) = 1$, and $\lambda_i \ge 0$ (Positive Semidefiniteness).
- Zero-pads non-power-of-2 open system states ($6\times 6 \to 8\times 8 \to 3$ qubits) or verifies strict mapping ($4\times 4 \to 2$ qubits).

### 4. Hybrid QML Continuous Parameter Regression
```
  [QuTiP Steady-State ρ_ss] 
             │
             ▼
  [Conversion Bridge] (Strict or 2^n Padded Embedding)
             │
             ▼
  [PennyLane QubitDensityMatrix] (default.mixed simulator)
             │
             ▼
  [Variational Circuit: RY / RZ + CNOT Entangler]
             │
             ▼
  [Expectation Values: <Z_0>, <Z_1>, <X_0>]
             │
             ▼
  [Classical Linear Head: w·f + b] ──► Estimated Angle θ_hat
```

---

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/your-username/Rydberg_Receiver_Simulations.git
cd Rydberg_Receiver_Simulations
pip install -r requirements.txt
```

### Running the Test Suite
Run the unified 5-part verification suite to validate all solvers, state conversions, gate stacks, and gradient flows:
```bash
python test_suite.py
```

### Running the QML Regression Pipeline
```bash
python qml_regression_demo.py --samples 150 --steps 60 --lr 0.05
```

### Running QFI / QCRB Parameter Sweeps
```bash
python qcrb_simulation_and_dataset.py
```

---

## 🛠 Tech Stack
- **Quantum Simulation:** QuTiP (Quantum Toolbox in Python)
- **Quantum Machine Learning:** PennyLane, Autograd
- **Scientific Computing:** NumPy, SciPy, SymPy (Wigner 3j / Clebsch-Gordan symbols)
- **Visualization:** Matplotlib, Jupyter