import numpy as np


def qutip_dm_to_pennylane(rho, *, check=True, renormalize=False, atol=1e-8):
    """
    Convert a density matrix into a PennyLane-ready qubit density matrix.

    Input:
      - rho: QuTiP Qobj or ndarray
      - check: enable physical validity checks (Hermitian + trace ~= 1)
      - renormalize: if True and trace != 1, divide matrix by its trace
      - atol: numerical tolerance used in checks

    Returns (rho_pl, n_qubits).
      - rho_pl: complex ndarray with shape (2^n, 2^n)
      - n_qubits: inferred number of qubits n
    """
    # QuTiP density matrices expose `.full()` to return a NumPy array.
    # If `rho` is already ndarray-like, this branch is skipped.
    if hasattr(rho, "full"):
        rho = rho.full()

    # Force a complex ndarray so downstream linear algebra is consistent.
    rho = np.asarray(rho, dtype=complex)

    # PennyLane state-preparation ops expect a 2D square matrix for density input.
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("density matrix must be a square 2D array")

    # Infer qubit count from matrix dimension. For qubit systems dim must be 2^n.
    dim = rho.shape[0]
    n_qubits = int(np.round(np.log2(dim)))
    if 2 ** n_qubits != dim:
        raise ValueError(f"dimension {dim} is not a power of 2 (qubits required)")

    if check:
        # Physical density matrices must be Hermitian: rho = rho^\dagger.
        if not np.allclose(rho, rho.conj().T, atol=atol):
            raise ValueError("density matrix must be Hermitian")

        # Physical density matrices must have unit trace.
        tr = np.trace(rho)
        if not np.allclose(tr, 1.0, atol=atol):
            if renormalize:
                # Optional repair path used for near-valid matrices.
                rho = rho / tr
            else:
                raise ValueError(f"density matrix trace is {tr}, expected 1")

    # Return plain ndarray + qubit count so caller can pass directly to PennyLane.
    return rho, n_qubits
