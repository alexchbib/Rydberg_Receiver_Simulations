import numpy as np
import qutip as qt
from qutip_pennylane_conversion import qutip_dm_to_pennylane


def H_RWA(Omega_p, Omega_c, Omega_RF, Delta_p, Delta_c, Delta_RF, hbar=1.0):
    H = (hbar / 2) * qt.Qobj([
        [0,                   Omega_p,                          0,                              0],
        [Omega_p,             -2*Delta_p,                       Omega_c,                        0],
        [0,                   Omega_c,                          -2* (Delta_p + Delta_c),              Omega_RF],
        [0,                   0,                                Omega_RF,                             -2*(Delta_p + Delta_c + Delta_RF)]
    ])
    return H


def decay_operators(gamma1, gamma2, gamma3, gamma4):
    c_ops = []
    if gamma2 > 0:
        c_ops.append(np.sqrt(gamma2) * qt.basis(4, 0) * qt.basis(4, 1).dag())
    if gamma3 > 0:
        c_ops.append(np.sqrt(gamma3) * qt.basis(4, 1) * qt.basis(4, 2).dag())
    if gamma4 > 0:
        c_ops.append(np.sqrt(gamma4) * qt.basis(4, 2) * qt.basis(4, 3).dag())
    return c_ops


def test_basic_conversion():
    H = H_RWA(0.04, 0.67, 0.8067, 0, 17.5, 0)
    c_ops = decay_operators(0, 5.2, 3.9, 0.17)
    rho_ss = qt.steadystate(H, c_ops, method="direct", tol=1e-12)

    rho_pl, n_qubits = qutip_dm_to_pennylane(rho_ss, check=True, renormalize=False, atol=1e-8)

    assert n_qubits == 2
    assert rho_pl.shape == (4, 4)
    assert np.allclose(np.trace(rho_pl), 1.0, atol=1e-8)
    assert np.allclose(rho_pl, rho_pl.conj().T, atol=1e-8)


def test_small_sweep():
    H_base = H_RWA(0.04, 0.67, 0.8067, 0, 0, 0)
    c_ops = decay_operators(0, 5.2, 3.9, 0.17)

    for Dc in np.linspace(-5, 5, 51):
        H = H_RWA(0.04, 0.67, 0.8067, 0, Dc, 0)
        rho_ss = qt.steadystate(H, c_ops, method="direct", tol=1e-12)
        rho_pl, _ = qutip_dm_to_pennylane(rho_ss, check=True, renormalize=False, atol=1e-8)
        assert np.allclose(np.trace(rho_pl), 1.0, atol=1e-8)


def test_renormalize():
    rho = np.eye(2)
    rho_pl, n_qubits = qutip_dm_to_pennylane(rho, check=True, renormalize=True)
    assert n_qubits == 1
    assert np.allclose(np.trace(rho_pl), 1.0, atol=1e-12)


def run_all():
    test_basic_conversion()
    test_small_sweep()
    test_renormalize()
    print("All conversion tests passed.")


if __name__ == "__main__":
    run_all()
