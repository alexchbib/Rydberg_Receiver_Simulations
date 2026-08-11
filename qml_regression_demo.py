"""
Variational Quantum Machine Learning (QML) Continuous Parameter Regression.

Role: QML Architecture, Variational Circuit Design, & Training Pipeline
Author / Implementation: Quantum Software Pipeline

Pipeline Flow:
  1. Open Quantum Dynamics: QuTiP 4-level steady-state density matrix rho_ss (4x4)
  2. State Bridge: Convert and validate rho_ss -> PennyLane 2-qubit state
  3. Variational Circuit (QNode): default.mixed mixed-state circuit with RY/RZ + CNOT
  4. Quantum Feature Extraction: Measure observables <Z0>, <Z1>, <X0>
  5. Hybrid Head: Classical linear readout (w . features + b)
  6. End-to-End Optimization: Autograd gradient validation + Adam optimizer
"""

import argparse
import importlib.metadata
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

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

# Theoretical 4-level Rydberg Hamiltonian & decay channels
from physics_models.four_level_model import H_RWA, decay_operators

# State conversion and physical validity verification bridge
from qutip_pennylane_conversion import qutip_dm_to_pennylane


def generate_pennylane_dataset(n_samples=150, seed=7):
    """
    Generate a 4x4 PennyLane-ready dataset from the 4-level simulation.

    Each sample:
      - X: 4x4 complex density matrix (maps to 2 qubits)
      - y: continuous RF angle target (theta_rf in radians)
    """
    rng = np.random.default_rng(seed)
    x = np.zeros((n_samples, 4, 4), dtype=np.complex128)
    y = np.zeros((n_samples,), dtype=np.float64)

    # Fixed dissipation channels for the 4-level system
    c_ops = decay_operators(0.0, 5.2, 3.9, 0.17)

    for i in range(n_samples):
        theta_rf = rng.uniform(0.0, np.pi / 2)
        delta_c = rng.uniform(-30.0, 30.0)

        # Imprint angle into RF Rabi coupling
        omega_rf = 0.8067 * np.cos(theta_rf)

        # Solve steady-state master equation: L(rho) = 0
        h = H_RWA(
            Omega_p=0.04,
            Omega_c=0.67,
            Omega_RF=omega_rf,
            Delta_p=0.0,
            Delta_c=delta_c,
            Delta_RF=0.0,
        )
        rho_ss = qt.steadystate(h, c_ops, method="direct", tol=1e-12)

        # Convert to 2-qubit PennyLane density matrix with validation
        rho_pl, n_qubits = qutip_dm_to_pennylane(rho_ss, check=True, renormalize=False)
        if n_qubits != 2:
            raise RuntimeError(f"Expected 2 qubits for 4x4 path, got {n_qubits}")

        x[i] = rho_pl
        y[i] = theta_rf

    return x, y


def build_quantum_feature_qnode():
    """
    Build a mixed-state Variational Quantum Circuit QNode returning 3 expectation values.

    Ansatz:
      - State Injection: QubitDensityMatrix(rho)
      - Parameterized Rotations: RY(theta_0), RZ(phi_0) on Wire 0; RY(theta_1), RZ(phi_1) on Wire 1
      - Entangler: CNOT(wire 0 -> wire 1)
      - Readout: <Z0>, <Z1>, <X0>
    """
    dev = qml.device("default.mixed", wires=2)

    @qml.qnode(dev, interface="autograd")
    def feature_qnode(rho, q_params):
        qml.QubitDensityMatrix(rho, wires=[0, 1])

        # Variational single-qubit rotations
        qml.RY(q_params[0, 0], wires=0)
        qml.RZ(q_params[0, 1], wires=0)
        qml.RY(q_params[1, 0], wires=1)
        qml.RZ(q_params[1, 1], wires=1)

        # Entanglement
        qml.CNOT(wires=[0, 1])

        # Quantum feature observables
        return (
            qml.expval(qml.PauliZ(0)),
            qml.expval(qml.PauliZ(1)),
            qml.expval(qml.PauliX(0)),
        )

    return feature_qnode


def predict_single(rho, q_params, head_w, head_b, feature_qnode):
    """Predict angle for one density matrix: y_hat = w . features + b."""
    features = pnp.stack(feature_qnode(rho, q_params))
    y_hat = pnp.dot(head_w, features) + head_b
    return y_hat


def mse_loss(q_params, head_w, head_b, x_data, y_data, feature_qnode):
    """Mean squared error loss over the training set."""
    preds = [predict_single(rho, q_params, head_w, head_b, feature_qnode) for rho in x_data]
    preds = pnp.stack(preds)
    return pnp.mean((preds - y_data) ** 2)


def predict_dataset(x_data, q_params, head_w, head_b, feature_qnode):
    """Run inference over a dataset and return NumPy predictions."""
    preds = np.zeros((len(x_data),), dtype=np.float64)
    for i, rho in enumerate(x_data):
        preds[i] = float(predict_single(rho, q_params, head_w, head_b, feature_qnode))
    return preds


def mae(y_true, y_pred):
    """Mean absolute error helper."""
    return float(np.mean(np.abs(y_pred - y_true)))


def train_model(x_train, y_train, steps=60, lr=0.05, seed=7):
    """
    Train quantum parameters + classical head jointly using Adam.
    Returns: learned parameters, loss history, and gradient norm checks.
    """
    rng = np.random.default_rng(seed)
    feature_qnode = build_quantum_feature_qnode()

    # Trainable parameters
    q_params = pnp.array(0.01 * rng.normal(size=(2, 2)), requires_grad=True)
    head_w = pnp.array(0.01 * rng.normal(size=(3,)), requires_grad=True)
    head_b = pnp.array(0.0, requires_grad=True)
    y_train = pnp.array(y_train, requires_grad=False)

    opt = qml.AdamOptimizer(stepsize=lr)

    def objective(qp, hw, hb):
        return mse_loss(qp, hw, hb, x_train, y_train, feature_qnode)

    initial_loss = float(objective(q_params, head_w, head_b))

    # Explicit gradient check proves end-to-end differentiability:
    # quantum params + classical head both receive gradients.
    grads = qml.grad(objective, argnums=[0, 1, 2])(q_params, head_w, head_b)
    grad_norm_q = float(np.linalg.norm(np.asarray(grads[0])))
    grad_norm_w = float(np.linalg.norm(np.asarray(grads[1])))
    grad_norm_b = float(np.linalg.norm(np.asarray(grads[2])))

    losses = [initial_loss]

    for step in range(1, steps + 1):
        q_params, head_w, head_b = opt.step(objective, q_params, head_w, head_b)
        loss_val = float(objective(q_params, head_w, head_b))
        losses.append(loss_val)

        if step == 1 or step % 10 == 0 or step == steps:
            print(f"Step {step:3d} | MSE Loss: {loss_val:.6f}")

    return (
        q_params,
        head_w,
        head_b,
        losses,
        feature_qnode,
        initial_loss,
        grad_norm_q,
        grad_norm_w,
        grad_norm_b,
    )


def plot_results(losses, y_true, y_pred):
    """Plot training loss curve and true vs. predicted regression scatter."""
    plt.figure(figsize=(12, 5))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(losses, linewidth=2, color="#1f77b4")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.title("QML Training Loss Curve")
    plt.grid(True, alpha=0.3)

    # Scatter plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, y_pred, alpha=0.75, color="#2ca02c", edgecolors="k", s=35)
    y_min = min(float(np.min(y_true)), float(np.min(y_pred)))
    y_max = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([y_min, y_max], [y_min, y_max], "r--", linewidth=1.5, label="Ideal y=x")
    plt.xlabel("True Angle (rad)")
    plt.ylabel("Predicted Angle (rad)")
    plt.title("Predicted vs True RF Angle")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Variational QML Continuous Angle Regression Demo.")
    parser.add_argument("--samples", type=int, default=150, help="Total dataset size.")
    parser.add_argument("--steps", type=int, default=60, help="Adam training iterations.")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--train-frac", type=float, default=0.8, help="Train split fraction.")
    args = parser.parse_args()

    print("=== Rydberg QML Parameter Estimation Pipeline ===")
    print("Generating 4-level open-system density matrix dataset...")
    x, y = generate_pennylane_dataset(n_samples=args.samples, seed=args.seed)
    print(f"Dataset generated: X={x.shape}, y={y.shape}")

    split_idx = int(args.train_frac * len(x))
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"Train samples: {len(x_train)}, Test samples: {len(x_test)}\n")

    (
        q_params,
        head_w,
        head_b,
        losses,
        feature_qnode,
        initial_loss,
        grad_norm_q,
        grad_norm_w,
        grad_norm_b,
    ) = train_model(x_train, y_train, steps=args.steps, lr=args.lr, seed=args.seed)

    final_loss = float(losses[-1])
    y_pred = predict_dataset(x_test, q_params, head_w, head_b, feature_qnode)
    test_mae = mae(y_test, y_pred)

    print("\n=== Training Summary ===")
    print(f"Initial MSE Loss: {initial_loss:.6f}")
    print(f"Final MSE Loss:   {final_loss:.6f}")
    print(f"Held-out Test MAE: {test_mae:.6f} rad")

    print("\n=== Gradient Flow Verification (at init) ===")
    print(f"||dL/dq_params|| = {grad_norm_q:.6e}")
    print(f"||dL/dhead_w||   = {grad_norm_w:.6e}")
    print(f"||dL/dhead_b||   = {grad_norm_b:.6e}")
    if grad_norm_q > 0 and grad_norm_w > 0:
        print("Status: Gradient flow healthy across quantum and classical layers.")

    plot_results(losses, y_test, y_pred)


if __name__ == "__main__":
    main()
