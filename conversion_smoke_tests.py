import numpy as np
import qutip as qt
from qutip_pennylane_conversion import qutip_dm_to_pennylane


def H_RWA(Omega_p, Omega_c, Omega_RF, Delta_p, Delta_c, Delta_RF, hbar=1.0):
    """
    Build the 4-level rotating-wave Hamiltonian used in the starting-point model.

    The basis ordering is fixed by the original scripts and the Hamiltonian
    follows the same sign/scale conventions so tests match production behavior.
    """
    H = (hbar / 2) * qt.Qobj([
        [0,                   Omega_p,                          0,                              0],
        [Omega_p,             -2*Delta_p,                       Omega_c,                        0],
        [0,                   Omega_c,                          -2* (Delta_p + Delta_c),              Omega_RF],
        [0,                   0,                                Omega_RF,                             -2*(Delta_p + Delta_c + Delta_RF)]
    ])
    return H


def decay_operators(gamma1, gamma2, gamma3, gamma4):
    """
    Build Lindblad collapse operators for the cascade decay chain:
      |2> -> |1>, |3> -> |2>, |4> -> |3>.
    """
    c_ops = []
    # Include each channel only when its decay rate is non-zero.
    if gamma2 > 0:
        c_ops.append(np.sqrt(gamma2) * qt.basis(4, 0) * qt.basis(4, 1).dag())
    if gamma3 > 0:
        c_ops.append(np.sqrt(gamma3) * qt.basis(4, 1) * qt.basis(4, 2).dag())
    if gamma4 > 0:
        c_ops.append(np.sqrt(gamma4) * qt.basis(4, 2) * qt.basis(4, 3).dag())
    return c_ops


def test_basic_conversion():
    """
    Single-point end-to-end check:
      simulation steady-state -> conversion -> physics + shape assertions.
    """
    # Use one representative operating point from the baseline simulation.
    H = H_RWA(0.04, 0.67, 0.8067, 0, 17.5, 0)
    c_ops = decay_operators(0, 5.2, 3.9, 0.17)
    rho_ss = qt.steadystate(H, c_ops, method="direct", tol=1e-12)

    # Convert QuTiP Qobj into PennyLane-ready ndarray.
    rho_pl, n_qubits = qutip_dm_to_pennylane(rho_ss, check=True, renormalize=False, atol=1e-8)

    # Verify qubit mapping and density-matrix validity.
    assert n_qubits == 2
    assert rho_pl.shape == (4, 4)
    assert np.allclose(np.trace(rho_pl), 1.0, atol=1e-8)
    assert np.allclose(rho_pl, rho_pl.conj().T, atol=1e-8)


def test_small_sweep():
    """
    Sweep detuning values to ensure conversion remains stable across conditions.
    """
    c_ops = decay_operators(0, 5.2, 3.9, 0.17)

    for Dc in np.linspace(-5, 5, 51):
        # Recompute steady-state for each sweep point and convert it.
        H = H_RWA(0.04, 0.67, 0.8067, 0, Dc, 0)
        rho_ss = qt.steadystate(H, c_ops, method="direct", tol=1e-12)
        rho_pl, _ = qutip_dm_to_pennylane(rho_ss, check=True, renormalize=False, atol=1e-8)
        # Minimal invariant checked for every point: trace remains normalized.
        assert np.allclose(np.trace(rho_pl), 1.0, atol=1e-8)


def test_renormalize():
    """
    Verify optional trace-fix behavior in the converter.
    """
    # Identity(2) has trace 2, so this intentionally requires renormalization.
    rho = np.eye(2)
    rho_pl, n_qubits = qutip_dm_to_pennylane(rho, check=True, renormalize=True)
    assert n_qubits == 1
    assert np.allclose(np.trace(rho_pl), 1.0, atol=1e-12)


def run_all():
    """Run all smoke tests in a deterministic order."""
    test_basic_conversion()
    test_small_sweep()
    test_renormalize()
    print("All conversion tests passed.")


if __name__ == "__main__":
    run_all()
