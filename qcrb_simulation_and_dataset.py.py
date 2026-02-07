import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, basis, steadystate
from qutip_pennylane_conversion import qutip_dm_to_pennylane


# Physical constants
HBAR = 1.054571817e-34
MU_B = 9.274009994e-24
E_CHARGE = 1.60217662e-19
A0 = 5.29177210903e-11

# System constants
G_S = 2.0
G_P = 0.67
D_REDUCED = E_CHARGE * A0 * 1443.459

# Laser and RF defaults
OMEGA_P = 2 * np.pi * 5e6
OMEGA_C = 2 * np.pi * 1e6
F_RF = 6.9e9
OMEGA_0 = 2 * np.pi * F_RF
OMEGA_RF_DEFAULT = OMEGA_0 + 2 * np.pi * 0.8e06
DELTA_P = 0.0
DELTA_C = 0.0

# Dissipation defaults
GAMMA2 = 2 * np.pi * 6.67e6
GAMMA3 = 2 * np.pi * 5e3
GAMMA4 = 2 * np.pi * 3e3
GAMMA_34 = (GAMMA3 + GAMMA4) / 2


TRANSITIONS = [
    {"mJp": +0.5, "mJ": +0.5, "q": 0, "CG": 1 / np.sqrt(6), "phase": +1},
    {"mJp": -0.5, "mJ": -0.5, "q": 0, "CG": 1 / np.sqrt(6), "phase": -1},
    {"mJp": -0.5, "mJ": +0.5, "q": +1, "CG": 1 / np.sqrt(6), "phase": +1},
    {"mJp": +0.5, "mJ": -0.5, "q": -1, "CG": 1 / np.sqrt(6), "phase": -1},
]

LABELS = ["g", "e", "r+", "r-", "p+", "p-"]
STATES = {name: basis(6, i) for i, name in enumerate(LABELS)}
PROJ = {k: v * v.dag() for k, v in STATES.items()}

SIG_GE = STATES["g"] * STATES["e"].dag()
SIG_EG = SIG_GE.dag()
SIG_ERP = STATES["r+"] * STATES["e"].dag()
SIG_RPE = SIG_ERP.dag()
SIG_ERM = STATES["r-"] * STATES["e"].dag()
SIG_RME = SIG_ERM.dag()

CG_CP_P = 1 / np.sqrt(2)
CG_CP_M = 1 / np.sqrt(2)


def alpha_pol(theta, q):
    if q == +1:
        return -1 / np.sqrt(2) * np.sin(theta)
    if q == -1:
        return 1 / np.sqrt(2) * np.sin(theta)
    if q == 0:
        return np.cos(theta)
    return 0.0


def omega_rf_components(theta_rf, theta_b, e0, b_field, omega_rf):
    components = []
    for tr in TRANSITIONS:
        delta_z = (G_P * tr["mJ"] - G_S * tr["mJp"]) * MU_B * b_field * np.cos(theta_b) / HBAR
        delta_q = omega_rf - (OMEGA_0 + delta_z)
        omega_q = (e0 / HBAR) * abs(alpha_pol(theta_rf, tr["q"]) * tr["phase"] * tr["CG"] * D_REDUCED)
        components.append((omega_q, delta_q, tr["mJp"], tr["mJ"]))
    return components


def h_system(theta_rf, theta_b, e0, b_field, omega_rf=OMEGA_RF_DEFAULT):
    h_p = (OMEGA_P / 2) * (SIG_GE + SIG_EG)
    h_c = (OMEGA_C / 2) * (CG_CP_P * (SIG_ERP + SIG_RPE) + CG_CP_M * (SIG_ERM + SIG_RME))

    h_rf = Qobj(np.zeros((6, 6), dtype=complex))
    for omega_q, _, m_jp, m_j in omega_rf_components(theta_rf, theta_b, e0, b_field, omega_rf):
        idx_r = "r+" if m_jp == +0.5 else "r-"
        idx_p = "p+" if m_j == +0.5 else "p-"
        op_rp = STATES[idx_r] * STATES[idx_p].dag()
        h_rf += (omega_q / 2) * (op_rp + op_rp.dag())

    h_det = DELTA_P * PROJ["e"] + DELTA_C * (PROJ["r+"] + PROJ["r-"])
    for _, delta_q, _, m_j in omega_rf_components(theta_rf, theta_b, e0, b_field, omega_rf):
        key = "p+" if m_j == +0.5 else "p-"
        h_det += delta_q * PROJ[key]

    return h_p + h_c + h_rf + h_det


def collapse_operators():
    return [
        np.sqrt(GAMMA2) * SIG_GE,
        np.sqrt(GAMMA3) * (SIG_RPE + SIG_RME),
        np.sqrt(GAMMA4) * (STATES["r+"] * STATES["p+"].dag() + STATES["r-"] * STATES["p-"].dag()),
        np.sqrt(GAMMA_34) * PROJ["p+"],
        np.sqrt(GAMMA_34) * PROJ["p-"],
    ]


def rho_ss(theta_rf, theta_b, e0, b_field, omega_rf=OMEGA_RF_DEFAULT):
    return steadystate(h_system(theta_rf, theta_b, e0, b_field, omega_rf), collapse_operators())


def compute_qfi(theta_rf, theta_b, e0, b_field, eps=1e-5, omega_rf=OMEGA_RF_DEFAULT):
    rho0 = rho_ss(theta_rf, theta_b, e0, b_field, omega_rf)
    rho_p = rho_ss(theta_rf + eps, theta_b, e0, b_field, omega_rf)
    rho_m = rho_ss(theta_rf - eps, theta_b, e0, b_field, omega_rf)
    drho = (rho_p - rho_m) / (2 * eps)

    vec = lambda a: a.full().ravel(order="F")
    mat = np.kron(rho0.full().conj(), np.eye(6)) + np.kron(np.eye(6), rho0.full())
    mat_inv = np.linalg.inv(mat + 1e-12 * np.eye(36))
    v = vec(drho)
    return 2 * np.real(v.conj() @ mat_inv @ v)


def _pad_density_matrix_to_power_of_two(rho_array):
    # PennyLane qubit density matrices require 2^n dimensions.
    # The demo model is 6-level (6x6), so we optionally embed it into 8x8.
    dim = rho_array.shape[0]
    target_dim = 1 << int(np.ceil(np.log2(dim)))
    if target_dim == dim:
        return rho_array

    padded = np.zeros((target_dim, target_dim), dtype=complex)
    padded[:dim, :dim] = rho_array
    return padded


def to_pennylane_density_matrix(rho, atol=1e-8, mode="pad"):
    """
    Convert QuTiP or ndarray density matrix to PennyLane-ready ndarray.

    mode:
      - "pad": zero-pad to next power-of-two dimension (default).
      - "strict": require dimension already power-of-two.

    Why this exists:
      The base converter enforces qubit dimensions (2^n). This helper keeps
      conversion logic in one place and makes the 6x6 -> 8x8 behavior explicit.
    """
    rho_array = rho.full() if hasattr(rho, "full") else np.asarray(rho, dtype=complex)

    if mode == "pad":
        # Non-qubit dimensions are embedded before calling the shared converter.
        rho_array = _pad_density_matrix_to_power_of_two(rho_array)
        return qutip_dm_to_pennylane(rho_array, check=True, renormalize=False, atol=atol)

    if mode == "strict":
        return qutip_dm_to_pennylane(rho_array, check=True, renormalize=False, atol=atol)

    raise ValueError("mode must be 'pad' or 'strict'")


def sweep_resolution_vs_e0(theta_rf_deg=30.0, theta_b_deg=30.0, b_field=0.2e-4, nu=1e4):
    theta_rf = np.deg2rad(theta_rf_deg)
    theta_b = np.deg2rad(theta_b_deg)
    e0_values = np.logspace(-3, 0, 500)
    res = np.zeros_like(e0_values)
    qfi_cut = 5e-3

    for i, e0 in enumerate(e0_values):
        f_val = compute_qfi(theta_rf, theta_b, e0=e0, b_field=b_field, eps=1e-5)
        res[i] = np.nan if f_val < qfi_cut else np.degrees(np.sqrt(1.0 / (nu * f_val)))

    return e0_values, res


def sweep_resolution_vs_b(theta_rf_deg=30.0, theta_b_deg=30.0, e0=0.1, nu=1e4):
    theta_rf = np.deg2rad(theta_rf_deg)
    theta_b = np.deg2rad(theta_b_deg)
    b_values = np.logspace(-5, -2, 100)
    res = np.zeros_like(b_values)

    for i, b_field in enumerate(b_values):
        f_val = compute_qfi(theta_rf, theta_b, e0=e0, b_field=b_field, eps=1e-5)
        res[i] = np.degrees(np.sqrt(1.0 / (nu * f_val)))

    return b_values, res


def generate_dataset(n_samples=500, seed=42):
    rng = np.random.default_rng(seed)
    x = np.zeros((n_samples, 6, 6), dtype=np.complex128)
    y = np.zeros((n_samples,), dtype=np.float64)

    for i in range(n_samples):
        theta_rf = rng.uniform(0.0, np.pi / 2)
        theta_b = rng.uniform(0.0, np.pi / 2)
        e0 = rng.uniform(1e-3, 1.0)
        b_field = 10 ** rng.uniform(-5, -2)

        rho = rho_ss(theta_rf, theta_b, e0=e0, b_field=b_field)
        x[i] = rho.full()
        y[i] = theta_rf

    return x, y


def generate_pennylane_dataset(n_samples=500, seed=42, conversion_mode="pad"):
    rng = np.random.default_rng(seed)

    # Probe one sample first to allocate output tensors with the converted size
    # (e.g., 8x8 when conversion_mode="pad").
    sample_rho = rho_ss(np.pi / 6, np.pi / 6, e0=0.1, b_field=1e-4)
    sample_pl, n_qubits = to_pennylane_density_matrix(sample_rho, mode=conversion_mode)
    dim = sample_pl.shape[0]

    x = np.zeros((n_samples, dim, dim), dtype=np.complex128)
    y = np.zeros((n_samples,), dtype=np.float64)

    for i in range(n_samples):
        theta_rf = rng.uniform(0.0, np.pi / 2)
        theta_b = rng.uniform(0.0, np.pi / 2)
        e0 = rng.uniform(1e-3, 1.0)
        b_field = 10 ** rng.uniform(-5, -2)

        rho = rho_ss(theta_rf, theta_b, e0=e0, b_field=b_field)
        rho_pl, _ = to_pennylane_density_matrix(rho, mode=conversion_mode)
        x[i] = rho_pl
        y[i] = theta_rf

    # Returns PennyLane-ready density matrices and target angle labels.
    return x, y, n_qubits


if __name__ == "__main__":
    e0_values, res_vs_e0 = sweep_resolution_vs_e0()
    plt.figure()
    plt.plot(e0_values, res_vs_e0, "-o")
    plt.xscale("log")
    plt.xlabel("Electric field amplitude E0 (V/m)")
    plt.ylabel("Angular resolution (deg)")
    plt.title("Resolution vs E0 at theta_RF=30 deg, theta_B=30 deg")
    plt.grid(True)
    plt.show()

    b_values, res_vs_b = sweep_resolution_vs_b()
    plt.figure()
    plt.plot(b_values * 1e3, res_vs_b, "-o")
    plt.xscale("log")
    plt.xlabel("Magnetic field magnitude B (mT)")
    plt.ylabel("Angular resolution (deg)")
    plt.title("Resolution vs B at theta_RF=30 deg, theta_B=30 deg, E0=0.1 V/m")
    plt.grid(True)
    plt.show()
