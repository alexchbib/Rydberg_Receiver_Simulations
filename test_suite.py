"""
Unified Test Suite for Rydberg Receiver Simulations & QML Pipeline.

Role: Automated Verification & Unit Testing Framework
Author / Implementation: Quantum Software Pipeline

Test Coverage:
  1. State Conversion Bridge: Hermiticity, Unit Trace, Positive Semidefiniteness (PSD)
  2. Sweep Stability: Numerical stability of QuTiP solvers across detunings
  3. Renormalization: Handling and repair of non-unit trace density matrices
  4. PennyLane Mixed-State Evolution: Gate stack execution on 2-qubit (4x4) and 3-qubit (8x8 padded) states
  5. QML Gradient Flow: Verifies non-vanishing autograd gradients across variational layers
"""

import importlib.metadata
import numpy as np

# Safe fallback for packaging/importlib.metadata in Anaconda environments
_orig_meta_version = importlib.metadata.version
def _safe_metadata_version(pkg):
    try:
        v = _orig_meta_version(pkg)
        if v is not None:
            return str(v)
    except Exception:
        pass
    if pkg == "numpy":
        return getattr(np, "__version__", "1.26.4")
    return "1.0.0"
importlib.metadata.version = _safe_metadata_version

import pennylane as qml
from pennylane import numpy as pnp
import qutip as qt

from physics_models.four_level_model import H_RWA, decay_operators
from qutip_pennylane_conversion import qutip_dm_to_pennylane
from qcrb_simulation_and_dataset import rho_ss_6level, to_pennylane_density_matrix
from qml_regression_demo import build_quantum_feature_qnode, mse_loss


def validate_density_matrix(rho, n_qubits, atol=1e-8):
    """Verify standard quantum mechanical density matrix properties."""
    dim = 2 ** n_qubits
    assert rho.shape == (dim, dim), f"Expected shape {(dim, dim)}, got {rho.shape}"
    assert np.allclose(rho, rho.conj().T, atol=atol), "Matrix is not Hermitian"
    assert np.allclose(np.trace(rho), 1.0, atol=atol), f"Trace is {np.trace(rho)}, expected 1.0"
    eigvals = np.linalg.eigvalsh(rho)
    assert np.min(eigvals) >= -1e-8, f"Matrix not PSD, min eigenvalue = {np.min(eigvals)}"


def test_basic_conversion():
    """Test 1: QuTiP steady-state to 2-qubit PennyLane density matrix conversion."""
    h = H_RWA(0.04, 0.67, 0.8067, 0.0, 17.5, 0.0)
    c_ops = decay_operators(0.0, 5.2, 3.9, 0.17)
    rho_ss = qt.steadystate(h, c_ops, method="direct", tol=1e-12)

    rho_pl, n_qubits = qutip_dm_to_pennylane(rho_ss, check=True, renormalize=False)

    assert n_qubits == 2
    assert rho_pl.shape == (4, 4)
    validate_density_matrix(rho_pl, n_qubits)
    print("  [PASS] Test 1: Basic 4x4 -> 2-qubit state conversion & validation")


def test_parameter_sweep_stability():
    """Test 2: Numerical solver stability across a continuous detuning sweep."""
    c_ops = decay_operators(0.0, 5.2, 3.9, 0.17)

    for dc in np.linspace(-5.0, 5.0, 31):
        h = H_RWA(0.04, 0.67, 0.8067, 0.0, dc, 0.0)
        rho_ss = qt.steadystate(h, c_ops, method="direct", tol=1e-12)
        rho_pl, n_qubits = qutip_dm_to_pennylane(rho_ss, check=True, renormalize=False)
        assert np.allclose(np.trace(rho_pl), 1.0, atol=1e-8)

    print("  [PASS] Test 2: Solver & conversion numerical stability across sweep")


def test_renormalization():
    """Test 3: Automatic trace repair path for near-valid matrices."""
    rho_unnormalized = np.eye(2, dtype=complex)  # trace = 2
    rho_pl, n_qubits = qutip_dm_to_pennylane(rho_unnormalized, check=True, renormalize=True)
    assert n_qubits == 1
    assert np.allclose(np.trace(rho_pl), 1.0, atol=1e-12)
    print("  [PASS] Test 3: Trace renormalization and error recovery")


def test_pennylane_gate_execution():
    """Test 4: Quantum gate evolution on both 4x4 (2-qubit) and 6x6 padded (3-qubit) states."""
    # 4x4 State
    h = H_RWA(0.04, 0.67, 0.8067, 0.0, 17.5, 0.0)
    c_ops = decay_operators(0.0, 5.2, 3.9, 0.17)
    rho4_ss = qt.steadystate(h, c_ops, method="direct", tol=1e-12)
    rho4, n4 = qutip_dm_to_pennylane(rho4_ss, check=True)

    dev2 = qml.device("default.mixed", wires=n4)

    @qml.qnode(dev2)
    def circuit4():
        qml.QubitDensityMatrix(rho4, wires=range(n4))
        qml.RX(0.25, wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.density_matrix(wires=range(n4))

    rho4_evolved = np.asarray(circuit4(), dtype=complex)
    validate_density_matrix(rho4_evolved, n4)
    assert np.linalg.norm(rho4_evolved - rho4) > 1e-10

    # 6x6 -> 8x8 Padded State
    rho6_ss = rho_ss_6level(np.pi / 6, np.pi / 6, e0=0.1, b_field=1e-4)
    rho8, n8 = to_pennylane_density_matrix(rho6_ss, mode="pad")
    assert n8 == 3
    assert rho8.shape == (8, 8)

    dev3 = qml.device("default.mixed", wires=n8)

    @qml.qnode(dev3)
    def circuit8():
        qml.QubitDensityMatrix(rho8, wires=range(n8))
        qml.RX(0.123, wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RZ(0.456, wires=2)
        return qml.density_matrix(wires=range(n8))

    rho8_evolved = np.asarray(circuit8(), dtype=complex)
    validate_density_matrix(rho8_evolved, n8)
    assert np.linalg.norm(rho8_evolved - rho8) > 1e-10

    print("  [PASS] Test 4: PennyLane gate execution on 2-qubit and 3-qubit padded states")


def test_qml_gradient_flow():
    """Test 5: Gradient flow across variational quantum circuit and classical readout head."""
    feature_qnode = build_quantum_feature_qnode()
    # Use a physical non-maximally mixed steady state so rotations produce state changes
    h = H_RWA(0.04, 0.67, 0.8067, 0.0, 17.5, 0.0)
    c_ops = decay_operators(0.0, 5.2, 3.9, 0.17)
    rho_ss = qt.steadystate(h, c_ops, method="direct", tol=1e-12)
    sample_rho, _ = qutip_dm_to_pennylane(rho_ss, check=True)

    q_params = pnp.array(np.random.normal(size=(2, 2)), requires_grad=True)
    head_w = pnp.array(np.random.normal(size=(3,)), requires_grad=True)
    head_b = pnp.array(0.0, requires_grad=True)

    x_dummy = [sample_rho]
    y_dummy = pnp.array([0.5], requires_grad=False)

    def objective(qp, hw, hb):
        return mse_loss(qp, hw, hb, x_dummy, y_dummy, feature_qnode)

    grads = qml.grad(objective, argnums=[0, 1, 2])(q_params, head_w, head_b)
    grad_norm_q = float(np.linalg.norm(np.asarray(grads[0])))
    grad_norm_w = float(np.linalg.norm(np.asarray(grads[1])))

    assert grad_norm_q > 0.0, f"Quantum circuit gradient is zero: {grad_norm_q}"
    assert grad_norm_w > 0.0, f"Classical readout gradient is zero: {grad_norm_w}"
    print("  [PASS] Test 5: End-to-end QML autograd differentiability & gradient flow")


def run_all():
    print("==================================================")
    print("Running Rydberg Receiver & QML Verification Suite")
    print("==================================================")
    test_basic_conversion()
    test_parameter_sweep_stability()
    test_renormalization()
    test_pennylane_gate_execution()
    test_qml_gradient_flow()
    print("==================================================")
    print("ALL TESTS PASSED (5/5) - System is verified.")
    print("==================================================")


if __name__ == "__main__":
    run_all()
