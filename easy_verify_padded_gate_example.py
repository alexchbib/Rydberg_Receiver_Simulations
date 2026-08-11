import numpy as np
import pennylane as qml

from qcrb_simulation_and_dataset import _pad_density_matrix_to_power_of_two
from qutip_pennylane_conversion import qutip_dm_to_pennylane


def main():
    # 1) Start with an easy 6x6 density matrix: pure population in first level.
    #    This is rho6 = |0><0| in the 6-level basis.
    rho6 = np.zeros((6, 6), dtype=complex)
    rho6[0, 0] = 1.0

    # 2) Pad to qubit size and validate with shared converter.
    rho8 = _pad_density_matrix_to_power_of_two(rho6)  # -> 8x8
    rho8_before, n_qubits = qutip_dm_to_pennylane(rho8, check=True, renormalize=False)

    # 3) Apply one simple gate in PennyLane: X on wire 0.
    dev = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(dev)
    def circuit():
        qml.QubitDensityMatrix(rho8_before, wires=range(n_qubits))
        qml.PauliX(wires=0)
        return qml.density_matrix(wires=range(n_qubits))

    rho8_after = np.asarray(circuit(), dtype=complex)

    # 4) Independent expected result: rho' = U rho U^\dagger.
    U = qml.matrix(qml.PauliX(wires=0), wire_order=range(n_qubits))
    rho8_expected = U @ rho8_before @ U.conj().T

    # 5) Easy verification checks.
    matches_expected = np.allclose(rho8_after, rho8_expected, atol=1e-10)
    trace_ok = np.allclose(np.trace(rho8_after), 1.0, atol=1e-12)
    herm_ok = np.allclose(rho8_after, rho8_after.conj().T, atol=1e-12)

    print("=== Easy Verification: 6x6 -> 8x8 + PauliX ===")
    print("input 6x6 shape:", rho6.shape)
    print("padded/input-to-PennyLane shape:", rho8_before.shape)
    print("output-after-gate shape:", rho8_after.shape)
    print("matches U rho U^dagger:", matches_expected)
    print("trace preserved:", trace_ok)
    print("Hermitian:", herm_ok)

    # Print small summaries for visual verification.
    print("\nNon-zero entries before gate:")
    nz_before = np.argwhere(np.abs(rho8_before) > 1e-12)
    for i, j in nz_before:
        print(f"rho_before[{i},{j}] = {rho8_before[i,j]}")

    print("\nNon-zero entries after gate:")
    nz_after = np.argwhere(np.abs(rho8_after) > 1e-12)
    for i, j in nz_after:
        print(f"rho_after[{i},{j}] = {rho8_after[i,j]}")

    if matches_expected and trace_ok and herm_ok:
        print("\nPASS: Gate application is correct and physically valid.")
    else:
        print("\nFAIL: Verification checks did not pass.")


if __name__ == "__main__":
    main()

