import numpy as np
import pennylane as qml

from qcrb_simulation_and_dataset import _pad_density_matrix_to_power_of_two
from qutip_pennylane_conversion import qutip_dm_to_pennylane


def print_matrix(label, mat, precision=3):
    """Pretty-print a complex matrix with fixed precision for visual comparison."""
    formatter = {"complex_kind": lambda z: f"{z.real:.{precision}f}{z.imag:+.{precision}f}j"}
    print(f"\n{label}")
    print(np.array2string(mat, formatter=formatter, max_line_width=200))


def apply_single_gate_to_6x6_density(rho6):
    """
    Demonstration helper for the exact workflow:
      6x6 density matrix -> 8x8 padded qubit matrix -> PennyLane gate evolution.

    Input:
      rho6: 6x6 density matrix (NumPy array).
            Expected to satisfy density-matrix properties:
              - square
              - Hermitian
              - trace = 1
              - positive semidefinite

    Output:
      rho8_before: 8x8 density matrix fed into PennyLane
      rho8_after:  8x8 density matrix after one gate (RX on wire 0)
      n_qubits:    inferred qubit count (3 for 8x8)
    """
    # Step 1: Convert user input into complex ndarray to keep numeric type stable
    # across linear-algebra ops and PennyLane interfaces.
    rho6 = np.asarray(rho6, dtype=complex)

    # Step 2: Pad 6x6 -> 8x8 so the state dimension is 2^n (required for qubits).
    # This embeds the original 6-level state into a larger Hilbert space.
    rho8 = _pad_density_matrix_to_power_of_two(np.asarray(rho6, dtype=complex))

    # Step 3: Run shared conversion checks and infer qubit count.
    # qutip_dm_to_pennylane validates square shape, Hermiticity, trace, and 2^n dim.
    # After padding, dim=8, so n_qubits=3.
    rho8_before, n_qubits = qutip_dm_to_pennylane(rho8, check=True, renormalize=False)

    # Step 4: Choose a mixed-state backend because we are evolving density matrices.
    dev = qml.device("default.mixed", wires=n_qubits)

    # Step 5: Define a QNode circuit that:
    #   (a) initializes from an external density matrix
    #   (b) applies one unitary gate
    #   (c) returns the resulting full density matrix
    @qml.qnode(dev)
    def circuit():
        # Initialize the full n-qubit state from rho8_before.
        qml.QubitDensityMatrix(rho8_before, wires=range(n_qubits))
        # Apply one example gate so the transformation is easy to explain.
        qml.RX(0.25, wires=0)
        # Read back the full evolved density matrix.
        return qml.density_matrix(wires=range(n_qubits))

    # Step 6: Execute the circuit and cast result to complex ndarray for consistency.
    rho8_after = np.asarray(circuit(), dtype=complex)
    return rho8_before, rho8_after, n_qubits


def main():
    # Example input: maximally mixed state on 6 levels.
    # This is a valid density matrix with trace 1 and diagonal entries 1/6.
    rho6 = np.eye(6, dtype=complex) / 6.0

    # Print original shape before any conversion/padding.
    print("original 6x6 input shape:", rho6.shape)

    # Run the full conversion + gate-application demonstration.
    rho8_before, rho8_after, n_qubits = apply_single_gate_to_6x6_density(rho6)

    # Print compact sanity checks for presentation/debugging.
    # - padded shape confirms 6x6 -> 8x8 mapping
    # - traces should remain ~1 under unitary gate evolution
    # - state_change_norm > 0 confirms the gate changed the state
    print("padded input shape:", rho8_before.shape)
    print("after-gate output shape:", rho8_after.shape)
    print("n_qubits:", n_qubits)
    print("trace before:", np.trace(rho8_before))
    print("trace after :", np.trace(rho8_after))
    print("state change norm:", np.linalg.norm(rho8_after - rho8_before))

    # Print full matrices so changes are visible entry-by-entry.
    print_matrix("original 6x6 density matrix", rho6)
    print_matrix("padded 8x8 density matrix before gate", rho8_before)
    print_matrix("8x8 density matrix after gate", rho8_after)


if __name__ == "__main__":
    # Script entry point: run the standalone demonstration.
    main()
