import numpy as np


def qutip_dm_to_pennylane(rho, *, check=True, renormalize=False, atol=1e-8):
    """
    Convert a QuTiP density matrix (Qobj or ndarray) into a PennyLane-ready ndarray.

    Returns (rho_pl, n_qubits).
    """
    if hasattr(rho, "full"):
        rho = rho.full()

    rho = np.asarray(rho, dtype=complex)

    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("density matrix must be a square 2D array")

    dim = rho.shape[0]
    n_qubits = int(np.round(np.log2(dim)))
    if 2 ** n_qubits != dim:
        raise ValueError(f"dimension {dim} is not a power of 2 (qubits required)")

    if check:
        if not np.allclose(rho, rho.conj().T, atol=atol):
            raise ValueError("density matrix must be Hermitian")

        tr = np.trace(rho)
        if not np.allclose(tr, 1.0, atol=atol):
            if renormalize:
                rho = rho / tr
            else:
                raise ValueError(f"density matrix trace is {tr}, expected 1")

    return rho, n_qubits
