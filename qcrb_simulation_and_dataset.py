"""
Quantum Metrology (QFI / QCRB) Simulation & Dataset Generation.

Role: Numerical Open-Systems Simulation & Metrology Limits Engine
Author / Implementation: Quantum Software Pipeline

Features:
  - Numerical steady-state master equation solver in QuTiP
  - Vectorized Safranek mixed-state Quantum Fisher Information (QFI) calculator
  - Quantum Cramér-Rao Bound (QCRB) parameter sweeps (Resolution vs E0, B)
  - Automated synthetic dataset generator for PennyLane QML models
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import steadystate
from qutip_pennylane_conversion import qutip_dm_to_pennylane

# Import theoretical domain models from physics_models package
from physics_models.six_level_zeeman_model import (
    h_system_6level,
    collapse_operators_6level,
    OMEGA_RF_DEFAULT,
)
from physics_models.four_level_model import H_RWA, decay_operators


#rho_ss: The Steady-State Density Matrix of the atom, computed by solving the Lindblad Master Equation using QuTiP.
def rho_ss_4level(Omega_p=0.04, Omega_c=0.67, Omega_RF=0.8067, Delta_p=0.0, Delta_c=0.0, Delta_RF=0.0):
    """Solve 4-level steady-state density matrix."""
    h = H_RWA(Omega_p, Omega_c, Omega_RF, Delta_p, Delta_c, Delta_RF)
    c_ops = decay_operators(0.0, 5.2, 3.9, 0.17)
    return steadystate(h, c_ops, method="direct", tol=1e-12)


"""
steadystate: is a function in qutip that solves the Lindblad Master Equation for a given Hamiltonian and collapse operators.
steadystate(h, c_ops) finds the exact equilibrium density matrix where the laser excitation perfectly balances spontaneous decay.
They return 4x4 or 6x6 matrices. Note: The diagonal numbers tell you what percentage of atoms are on each level at equilibrium, and the off-diagonal numbers tell you the laser coherences.
"""


def rho_ss_6level(theta_rf, theta_b, e0, b_field, omega_rf=OMEGA_RF_DEFAULT):
    """Solve 6-level Zeeman steady-state density matrix."""
    return steadystate(
        h_system_6level(theta_rf, theta_b, e0, b_field, omega_rf),
        collapse_operators_6level(),
    )


# Alias for backward compatibility
rho_ss = rho_ss_6level

"""
Comparison of 4-Level vs 6-Level Models:

- 4-Level Model:
  A simplified baseline where magnetic fields are ignored. It produces a 4x4 
  density matrix that maps directly to 2 qubits for our PennyLane QML regression.

- 6-Level Model:
  A realistic model that includes an external magnetic field. The magnetic field 
  splits the Rydberg levels into sublevels (r+, r-, p+, p-), allowing us to 
  simulate how accurately the atom can detect the angle of incoming radio waves.
"""




"""
Quantum Fisher Information (QFI) and the Quantum Cramér-Rao Bound (QCRB):

- QFI (F_Q):
  Measures how sensitive the atom's quantum state is to changes in the 
  incoming radio wave angle (theta_RF). Larger QFI = higher sensitivity.

- QCRB (Precision Limit):
  Sets the fundamental physical limit on measurement error (uncertainty):
  
      Uncertainty (Delta_theta) >= 1 / sqrt(nu * F_Q)
  
  Where: 
  - nu = number of measurements / repeats
  - F_Q = QFI (Quantum Fisher Information)

  A larger QFI means a smaller uncertainty, giving us a higher-resolution receiver.

"""


def compute_qfi(theta_rf, theta_b, e0, b_field, eps=1e-5, omega_rf=OMEGA_RF_DEFAULT):
    """

    Safranek Formula for Mixed-State QFI (Safranek 2018):

        QFI = 2 * (v_dagger @ inv(M) @ v)
    Where:
      - v = vec(d_rho / d_theta): 6x6 derivative matrix flattened into a 36-element vector.
      - M = (rho* ⊗ I) + (I ⊗ rho): 36x36 Kronecker super-matrix of the steady state.
      - inv(M): Inverted super-matrix (with 1e-12 on the diagonal to guarantee invertibility).
      - QFI: The final scalar sensitivity number.


    """
    rho0 = rho_ss_6level(theta_rf, theta_b, e0, b_field, omega_rf)
    rho_p = rho_ss_6level(theta_rf + eps, theta_b, e0, b_field, omega_rf)
    rho_m = rho_ss_6level(theta_rf - eps, theta_b, e0, b_field, omega_rf)

    # We simulate the atom at the normal angle (rho0), slightly tilted forward (rho_p), and slightly tilted backward (rho_m).

    # Numerical derivative d(rho)/d(theta_rf): Calculate how much the state changed
    drho = (rho_p - rho_m) / (2 * eps)


    # Vectorized Safranek Formula for Mixed-State QFI (Safranek 2018)
    # Formula: QFI = 2 * v_dagger @ inv(M) @ v
    
    
    # 1. Convert QuTiP objects to standard NumPy complex arrays
    rho_mat = rho0.full()       # 6x6 steady-state density matrix
    drho_mat = drho.full()      # 6x6 derivative matrix (d_rho / d_theta)

    # 2. Flatten the 6x6 derivative matrix into a 36-element vector v
    # (order="F" means column-by-column, the standard quantum physics convention)
    drho_vector = drho_mat.ravel(order="F")

    # 3. Build the 36x36 Kronecker super-matrix M = (rho* ⊗ I) + (I ⊗ rho)
    identity_6 = np.eye(6, dtype=complex)
    M = np.kron(rho_mat.conj(), identity_6) + np.kron(identity_6, rho_mat)

    # 4. Invert M (adding 1e-12 along the diagonal avoids dividing by zero): It's just an epsilon to prevent a division-by-zero crash in np.linalg.inv if the matrix contains zeros.
    identity_36 = np.eye(36, dtype=complex)
    M_inv = np.linalg.inv(M + 1e-12 * identity_36)
    # If your matrix M ever has det(M) = 0, adding 1e-12 to the diagonal guarantees that det(M) != 0, meaning the matrix is guaranteed to be 100% invertible and will not crash!
    
    # 5. Quadratic form: 2 * (v^dagger @ M_inv @ v) gives the scalar QFI value
    qfi_value = 2.0 * np.real(drho_vector.conj() @ M_inv @ drho_vector)
    
    return qfi_value



def _pad_density_matrix_to_power_of_two(rho_array):
    """
    Our Zeeman atom has 6 levels. But quantum computers cannot have "2.58 qubits", we must use 3 qubits (8 states).
    Subspace Embedding: Embeds a non-power-of-2 density matrix (e.g. 6x6) 
    into the next qubit Hilbert space dimension (8x8 -> 3 qubits).
    
    Why this is physically valid:
    - The original 6x6 atomic subspace is preserved exactly in the top-left block.
    - The extra dimensions (Levels 7 and 8) are set to zero probability.
    - Total trace remains exactly 1.0 (valid physical density matrix).
    """
    dim = rho_array.shape[0]  # shape[0] gives the first dimension of the matrix e.g. 6x6 has shape[0] = 6 and shape[1] = 6
    
    # 1. Number of qubits needed: ceil(log2(6)) = 3 qubits
    n_qubits = int(np.ceil(np.log2(dim)))
    
    # 2. Target matrix size: 2^3 = 8
    target_dim = 2 ** n_qubits 
    
    # If it's already a power of 2 (e.g. 4x4), no padding needed
    if target_dim == dim:
        return rho_array
    # 3. Create an 8x8 matrix of zeros and paste the 6x6 into the top-left
    padded = np.zeros((target_dim, target_dim), dtype=complex)
    padded[:dim, :dim] = rho_array
    return padded

def to_pennylane_density_matrix(rho, atol=1e-8, mode="pad"):
    """
    Convert QuTiP or NumPy density matrix to PennyLane qubit density matrix.

    Modes:
      - 'pad': zero-pad to next power-of-2 dimension (e.g. 6x6 -> 8x8 -> 3 qubits).
      - 'strict': require dimension already power-of-2 (e.g. 4x4 -> 2 qubits).
    """
    if hasattr(rho, "full"):
    # if the input 'rho' is a QuTiP density matrix object, convert it to a NumPy array
        rho_array = rho.full()
    else:
    # Ensure 'rho' is a complex NumPy array (casts lists and real arrays to complex)
        rho_array = np.asarray(rho, dtype=complex)


    # if the mode is 'pad', pad the density matrix to the next power-of-2 dimension
    if mode == "pad":
        rho_array = _pad_density_matrix_to_power_of_two(rho_array)
        return qutip_dm_to_pennylane(rho_array, check=True, renormalize=False, atol=atol)

    if mode == "strict":
        return qutip_dm_to_pennylane(rho_array, check=True, renormalize=False, atol=atol)

    raise ValueError("mode must be 'pad' or 'strict'")


"""
Why do we perform QCRB Resolution Sweeps?
1. Theoretical Baseline: Sets the ultimate physical precision limit to evaluate our QML regression model against.
2. Hardware Optimization: Determines the weakest detectable electric field (E0) and optimal magnetic bias (B).
3. Demonstration: Powers the performance curves in demo_QCRB_vs_E_and_B.ipynb.
"""

def sweep_resolution_vs_e0(theta_rf_deg=30.0, theta_b_deg=30.0, b_field=0.2e-4, nu=1e4):
    """Sweep electric field E0 from 0.001 to 1.0 V/m and compute angular resolution in degrees."""
    theta_rf = np.deg2rad(theta_rf_deg) # convert degrees to radians
    theta_b = np.deg2rad(theta_b_deg) # convert degrees to radians
    
    # 100 electric field points from 10^-3 (0.001 V/m) to 10^0 (1.0 V/m)
    e0_values = np.logspace(-3, 0, 100)

    res = np.zeros(len(e0_values)) # create an array of zeros with the same shape as e0_values
    # Calculate QFI and QCRB resolution for each electric field point

    for i, e0 in enumerate(e0_values): 
        qfi = compute_qfi(theta_rf, theta_b, e0=e0, b_field=b_field)
        # Apply QCRB formula: resolution = 1 / sqrt(nu * QFI)
        
        if qfi >= 5e-3:
            res[i] = np.degrees(np.sqrt(1.0 / (nu * qfi)))
        else:
            res[i] = np.nan # If QFI is near zero, signal is too weak to resolve
    return e0_values, res


def sweep_resolution_vs_b(theta_rf_deg=30.0, theta_b_deg=30.0, e0=0.1, nu=1e4):
    # Sweep magnetic field B from 10^-5 to 10^-2 Tesla (10 uT to 10 mT) and compute angular resolution in degrees.
    theta_rf = np.deg2rad(theta_rf_deg) # Convert degrees to radians
    theta_b = np.deg2rad(theta_b_deg)   # Convert degrees to radians
    
    # 100 magnetic field points from 10^-5 Tesla (10 microTesla) to 10^-2 Tesla (10 milliTesla)
    b_values = np.logspace(-5, -2, 100)
    res = np.zeros(len(b_values)) # create an array of zeros with the same shape as b_values
   
    # Calculate QFI and QCRB resolution for each magnetic field strength
    for i, b_field in enumerate(b_values):
        qfi = compute_qfi(theta_rf, theta_b, e0=e0, b_field=b_field)
        # Apply QCRB formula: resolution = 1 / sqrt(nu * QFI)
        res[i] = np.degrees(np.sqrt(1.0 / (nu * qfi)))
    return b_values, res

def generate_pennylane_6level_dataset(n_samples=200, seed=42, conversion_mode="pad"):
    """
    Synthetic Data Generator:
    Generates (X, y) training data from the 6-level simulation for PennyLane.
    
    Returns:
      - X: Array of 8x8 density matrices with shape (n_samples, 8, 8)
      - y: Array of true target angles theta_RF in radians with shape (n_samples,)
      - n_qubits: Inferred number of qubits (3 qubits for 8x8)
    """
    # initialize a random number generator with a specific seed
    rng = np.random.default_rng(seed)
    
    # Probe one sample to get the converted matrix dimension (dim = 8, n_qubits = 3), to determine the size of the matrix and allocate memory for X and y
    sample_rho = rho_ss_6level(np.pi / 6, np.pi / 6, e0=0.1, b_field=1e-4)
    
    # convert the 6x6 density matrix from QuTiP into a format PennyLane can use (an 8x8 matrix)
    sample_pl, n_qubits = to_pennylane_density_matrix(sample_rho, mode=conversion_mode)
    dim = sample_pl.shape[0] # 8
    
    # Pre-allocate empty arrays for X (inputs) and y (labels)
    X = np.zeros((n_samples, dim, dim), dtype=complex)
    y = np.zeros(n_samples, dtype=float)
    
    # Generate random operating points and simulate the atoms
    for i in range(n_samples):
        # Sample random target angle (0 to 90 degrees in radians)
        theta_rf = rng.uniform(0.0, np.pi / 2)
        
        # Sample random background noise parameters (angles & field strengths)
        theta_b = rng.uniform(0.0, np.pi / 2)
        e0 = rng.uniform(1e-3, 1.0)
        b_field = 10 ** rng.uniform(-5, -2)
        # Solve the atom in QuTiP and convert to PennyLane 8x8 density matrix
        rho = rho_ss_6level(theta_rf, theta_b, e0=e0, b_field=b_field)
        rho_pl, _ = to_pennylane_density_matrix(rho, mode=conversion_mode)
        
        # Store in dataset
        X[i] = rho_pl
        y[i] = theta_rf
    return X, y, n_qubits


if __name__ == "__main__":
    print("Running QCRB resolution sweeps...")
    e0_vals, res_e0 = sweep_resolution_vs_e0()
    print(f"Computed {len(e0_vals)} points for Resolution vs E0.")
    b_vals, res_b = sweep_resolution_vs_b()
    print(f"Computed {len(b_vals)} points for Resolution vs B.")
    print("Simulation complete.")
