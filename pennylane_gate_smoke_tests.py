import numpy as np
import pennylane as qml
import qutip as qt

from conversion_smoke_tests import H_RWA, decay_operators
from qutip_pennylane_conversion import qutip_dm_to_pennylane
from qcrb_simulation_and_dataset import rho_ss, to_pennylane_density_matrix


def validate_density_matrix(rho, n_qubits, atol=1e-8):
    """
    Validate that `rho` is a numerically valid qubit density matrix.
    Checks:
      - expected 2^n square shape
      - Hermiticity
      - unit trace
      - positive semidefinite (within small numeric tolerance)
    """
    dim = 2 ** n_qubits
    assert rho.shape == (dim, dim), f"unexpected shape {rho.shape}, expected {(dim, dim)}"
    assert np.allclose(rho, rho.conj().T, atol=atol), "density matrix is not Hermitian"
    assert np.allclose(np.trace(rho), 1.0, atol=atol), f"trace is {np.trace(rho)}"
    # PSD check via eigenvalues for Hermitian matrix.
    eigvals = np.linalg.eigvalsh(rho)
    assert np.min(eigvals) >= -1e-8, f"matrix is not PSD, min eigenvalue {np.min(eigvals)}"


def apply_basic_gates(rho, n_qubits):
    """
    Load density matrix into PennyLane and apply a small gate stack.

    This verifies the converted matrix is accepted by `QubitDensityMatrix`
    and can participate in mixed-state simulation.
    """
    # `default.mixed` supports density-matrix evolution.
    dev = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(dev)
    def circuit():
        # Initialize device with externally generated density matrix.
        qml.QubitDensityMatrix(rho, wires=range(n_qubits))
        # Apply simple parameterized and entangling operations.
        qml.RX(0.123, wires=0)
        if n_qubits > 1:
            qml.CNOT(wires=[0, 1])
            qml.RY(0.456, wires=1)
        if n_qubits > 2:
            qml.CZ(wires=[1, 2])
            qml.RZ(0.321, wires=2)
        # Return final density matrix for post-checks.
        return qml.density_matrix(wires=range(n_qubits))

    return np.asarray(circuit(), dtype=complex)


def build_4x4_sample():
    """
    Build one 4x4 steady-state sample from the starting-point 4-level model,
    then convert with the strict qubit-size converter.
    """
    h = H_RWA(0.04, 0.67, 0.8067, 0.0, 17.5, 0.0)
    c_ops = decay_operators(0.0, 5.2, 3.9, 0.17)
    rho_qutip = qt.steadystate(h, c_ops, method="direct", tol=1e-12)
    return qutip_dm_to_pennylane(rho_qutip, check=True, renormalize=False)


def build_6x6_padded_sample():
    """
    Build one 6x6 steady-state sample from the QCRB model,
    then convert using pad mode (6x6 -> 8x8 -> 3 qubits).
    """
    rho_qutip_6 = rho_ss(np.pi / 6, np.pi / 6, e0=0.1, b_field=1e-4)
    return to_pennylane_density_matrix(rho_qutip_6, mode="pad")


def run_all():
    """Execute both conversion paths and validate PennyLane gate compatibility."""
    # Path 1: strict qubit-sized converter (4x4 -> 2 qubits)
    rho4, n4 = build_4x4_sample()
    validate_density_matrix(rho4, n4)
    rho4_after = apply_basic_gates(rho4, n4)
    validate_density_matrix(rho4_after, n4)
    # Ensure state evolution is non-trivial.
    assert np.linalg.norm(rho4_after - rho4) > 1e-12, "4x4 state did not change after gates"

    # Path 2: padded converter (6x6 -> 8x8 -> 3 qubits)
    rho8, n8 = build_6x6_padded_sample()
    validate_density_matrix(rho8, n8)
    rho8_after = apply_basic_gates(rho8, n8)
    validate_density_matrix(rho8_after, n8)
    assert np.linalg.norm(rho8_after - rho8) > 1e-12, "8x8 state did not change after gates"

    print("PennyLane gate smoke tests passed.")
    print(f"4x4 path: input shape {rho4.shape}, n_qubits={n4}")
    print(f"6x6 padded path: input shape {rho8.shape}, n_qubits={n8}")


if __name__ == "__main__":
    run_all()
