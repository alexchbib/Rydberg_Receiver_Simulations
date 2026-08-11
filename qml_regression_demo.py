"""
Minimal proof-of-concept QML regression demo for the strict 4x4 path.

Pipeline shown in this script:
  QuTiP steady-state density matrix (4x4)
      -> strict converter (4x4 -> 2 qubits)
      -> PennyLane QNode feature extraction
      -> classical linear regression head
      -> angle prediction (continuous target)


"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import pennylane as qml
from pennylane import numpy as pnp

# Reuse existing validated 4-level simulation building blocks.
from conversion_smoke_tests import H_RWA, decay_operators
# Reuse strict converter (no padding). Ensures qubit-dimension safety checks.
from qutip_pennylane_conversion import qutip_dm_to_pennylane


def generate_pennylane_dataset(n_samples=150, seed=7):
    """
    Generate a strict 4x4 PennyLane-ready dataset from the existing 4-level simulation.

    Each sample:
      - rho: 4x4 complex density matrix (strict qubit path -> 2 qubits)
      - y:   continuous angle label (theta_rf in radians)

    Angle embedding used for this demo:
      Omega_RF = 0.8067 * cos(theta_rf)
    plus a random nuisance detuning Delta_c, so regression is non-trivial.
    """
    # Seed controls reproducibility: same seed -> same synthetic dataset.
    rng = np.random.default_rng(seed)

    # X stores one 4x4 density matrix per sample.
    x = np.zeros((n_samples, 4, 4), dtype=np.complex128)
    # y stores the regression target angle (continuous).
    y = np.zeros((n_samples,), dtype=np.float64)

    # Fixed dissipative channels from the starting-point model.
    c_ops = decay_operators(0.0, 5.2, 3.9, 0.17)

    for i in range(n_samples):
        # Continuous target angle in [0, pi/2].
        theta_rf = rng.uniform(0.0, np.pi / 2)
        # Nuisance parameter to avoid one-dimensional trivial fitting.
        delta_c = rng.uniform(-30.0, 30.0)

        # Angle-dependent RF coupling used to imprint target into rho.
        omega_rf = 0.8067 * np.cos(theta_rf)

        # Build Hamiltonian with sampled parameters.
        h = H_RWA(
            Omega_p=0.04,
            Omega_c=0.67,
            Omega_RF=omega_rf,
            Delta_p=0.0,
            Delta_c=delta_c,
            Delta_RF=0.0,
        )
        # QuTiP solves master equation steady state: L(rho)=0.
        rho_ss = qt.steadystate(h, c_ops, method="direct", tol=1e-12)

        # Strict conversion: must remain 4x4 and map to exactly 2 qubits.
        rho_pl, n_qubits = qutip_dm_to_pennylane(rho_ss, check=True, renormalize=False)
        if n_qubits != 2:
            raise RuntimeError(f"Expected 2 qubits for strict 4x4 path, got {n_qubits}")

        x[i] = rho_pl
        y[i] = theta_rf

    return x, y


def build_quantum_feature_qnode():
    """
    Build a tiny mixed-state QNode that returns 3 quantum features.

    Circuit:
      1) Load density matrix via QubitDensityMatrix
      2) Trainable single-qubit RY/RZ on both qubits
      3) One CNOT entangler
      4) Return <Z0>, <Z1>, <X0>
    """
    # Mixed-state simulator is required because inputs are density matrices.
    dev = qml.device("default.mixed", wires=2)

    @qml.qnode(dev, interface="autograd")
    def feature_qnode(rho, q_params):
        # Input state injection from simulation-converted density matrix.
        qml.QubitDensityMatrix(rho, wires=[0, 1])

        # Very small variational block: 2 rotations per qubit.
        qml.RY(q_params[0, 0], wires=0)
        qml.RZ(q_params[0, 1], wires=0)
        qml.RY(q_params[1, 0], wires=1)
        qml.RZ(q_params[1, 1], wires=1)

        # Single entangling operation (minimal ansatz).
        qml.CNOT(wires=[0, 1])

        # Three scalar observables become a compact learned feature vector.
        return (
            qml.expval(qml.PauliZ(0)),
            qml.expval(qml.PauliZ(1)),
            qml.expval(qml.PauliX(0)),
        )

    return feature_qnode


def predict_single(rho, q_params, head_w, head_b, feature_qnode):
    """
    Predict one angle:
      y_hat = w dot features + b
    """
    # QNode output is a tuple of 3 expectation values -> convert to vector.
    features = pnp.stack(feature_qnode(rho, q_params))
    # Classical linear readout maps quantum features to one scalar prediction.
    y_hat = pnp.dot(head_w, features) + head_b
    return y_hat


def mse_loss(q_params, head_w, head_b, x_data, y_data, feature_qnode):
    """
    Mean squared error over a dataset.
    No batching by design (simple loop for clarity).
    """
    preds = []
    for rho in x_data:
        # No vectorization/batching: deliberately explicit for readability.
        preds.append(predict_single(rho, q_params, head_w, head_b, feature_qnode))
    preds = pnp.stack(preds)
    # Standard regression objective.
    return pnp.mean((preds - y_data) ** 2)


def predict_dataset(x_data, q_params, head_w, head_b, feature_qnode):
    """Run inference on a full dataset and return NumPy predictions."""
    preds = np.zeros((len(x_data),), dtype=np.float64)
    for i, rho in enumerate(x_data):
        # Convert autograd scalar to plain float for reporting/plotting.
        preds[i] = float(predict_single(rho, q_params, head_w, head_b, feature_qnode))
    return preds


def mae(y_true, y_pred):
    """Mean absolute error helper for reporting."""
    return float(np.mean(np.abs(y_pred - y_true)))


def train_model(x_train, y_train, steps=60, lr=0.05, seed=7):
    """
    Train quantum parameters + linear head with Adam optimizer.
    Returns learned params, loss history, and gradient norms at initialization.
    """
    rng = np.random.default_rng(seed)
    feature_qnode = build_quantum_feature_qnode()

    # Trainable quantum parameters: row=qubit, col=gate type (RY then RZ).
    q_params = pnp.array(0.01 * rng.normal(size=(2, 2)), requires_grad=True)
    # Linear regression head parameters.
    head_w = pnp.array(0.01 * rng.normal(size=(3,)), requires_grad=True)
    head_b = pnp.array(0.0, requires_grad=True)

    # Ensure labels are compatible with autograd operations.
    y_train = pnp.array(y_train, requires_grad=False)

    opt = qml.AdamOptimizer(stepsize=lr)

    # Objective closure over fixed train set.
    def objective(qp, hw, hb):
        return mse_loss(qp, hw, hb, x_train, y_train, feature_qnode)

    # Baseline loss before any optimization.
    initial_loss = float(objective(q_params, head_w, head_b))

    # Explicit gradient check proves end-to-end differentiability:
    # quantum params + classical head both receive gradients.
    grads = qml.grad(objective, argnum=[0, 1, 2])(q_params, head_w, head_b)
    grad_norm_q = float(np.linalg.norm(np.asarray(grads[0])))
    grad_norm_w = float(np.linalg.norm(np.asarray(grads[1])))
    grad_norm_b = float(np.linalg.norm(np.asarray(grads[2])))

    losses = [initial_loss]

    for step in range(1, steps + 1):
        # One Adam update on all trainable parameter groups.
        q_params, head_w, head_b = opt.step(objective, q_params, head_w, head_b)
        # Re-evaluate full-train loss after the step.
        loss_val = float(objective(q_params, head_w, head_b))
        losses.append(loss_val)

        if step == 1 or step % 10 == 0 or step == steps:
            print(f"step {step:3d} | loss {loss_val:.6f}")

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
    """Create required plots: loss curve and prediction scatter."""
    plt.figure(figsize=(12, 5))

    # Plot 1: training objective trend (did optimization improve?).
    plt.subplot(1, 2, 1)
    plt.plot(losses, linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("MSE loss")
    plt.title("Training Loss vs Iteration")
    plt.grid(True, alpha=0.3)

    # Plot 2: prediction quality (ideal points lie on diagonal y=x).
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, y_pred, alpha=0.75)
    y_min = min(float(np.min(y_true)), float(np.min(y_pred)))
    y_max = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([y_min, y_max], [y_min, y_max], "r--", linewidth=1.5, label="Ideal y=x")
    plt.xlabel("True angle (rad)")
    plt.ylabel("Predicted angle (rad)")
    plt.title("Predicted vs True Angle")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Minimal QML regression demo (strict 4x4 path).")
    parser.add_argument("--samples", type=int, default=150, help="Total dataset size (100-500 recommended).")
    parser.add_argument("--steps", type=int, default=60, help="Training steps (50-100 recommended).")
    parser.add_argument("--lr", type=float, default=0.05, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument("--train-frac", type=float, default=0.8, help="Train split fraction.")
    args = parser.parse_args()

    print("Generating strict 4x4 PennyLane dataset...")
    x, y = generate_pennylane_dataset(n_samples=args.samples, seed=args.seed)
    print(f"Dataset shape: X={x.shape}, y={y.shape}")

    # Deterministic split for reproducible comparison between runs.
    split_idx = int(args.train_frac * len(x))
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"Train samples: {len(x_train)}, Test samples: {len(x_test)}")

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

    # Compute final metrics on held-out test set.
    final_loss = float(losses[-1])
    y_pred = predict_dataset(x_test, q_params, head_w, head_b, feature_qnode)
    test_mae = mae(y_test, y_pred)

    print("\n=== Training Summary ===")
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Final loss:   {final_loss:.6f}")
    print(f"Test MAE:     {test_mae:.6f} rad")

    # Explicit gradient-flow confirmation for presentation credibility.
    print("\n=== Gradient Flow Check (at initialization) ===")
    print(f"||dL/dq_params|| = {grad_norm_q:.6e}")
    print(f"||dL/dhead_w||   = {grad_norm_w:.6e}")
    print(f"||dL/dhead_b||   = {grad_norm_b:.6e}")
    if grad_norm_q > 0 and grad_norm_w > 0:
        print("Gradient flow status: OK (non-zero gradients detected).")
    else:
        print("Gradient flow status: CHECK (one or more gradients are zero).")

    plot_results(losses, y_test, y_pred)


if __name__ == "__main__":
    main()
