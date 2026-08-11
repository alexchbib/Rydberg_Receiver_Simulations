"""
Six-Level Zeeman-Resolved Rydberg Atomic Model.

Defines the Hamiltonian including fine structure / Zeeman magnetic sublevels:
|g>, |e>, |r+>, |r->, |p+>, |p->
accounting for Clebsch-Gordan coefficients, polarization projection,
and magnetic bias field angles.
"""

import numpy as np
from qutip import Qobj, basis

# Physical constants
HBAR = 1.054571817e-34       # J*s (reduced Planck constant)
MU_B = 9.274009994e-24       # J/T (Bohr magneton)
E_CHARGE = 1.60217662e-19    # C (elementary charge)
A0 = 5.29177210903e-11       # m (Bohr radius)

# Lande g-factors & dipole moment
G_S = 2.0
G_P = 0.67
D_REDUCED = E_CHARGE * A0 * 1443.459

# Laser and RF frequency defaults
OMEGA_P = 2 * np.pi * 5e6
OMEGA_C = 2 * np.pi * 1e6
F_RF = 6.9e9
OMEGA_0 = 2 * np.pi * F_RF
OMEGA_RF_DEFAULT = OMEGA_0 + 2 * np.pi * 0.8e06
DELTA_P = 0.0
DELTA_C = 0.0

# Dissipation & decay rates
GAMMA2 = 2 * np.pi * 6.67e6
GAMMA3 = 2 * np.pi * 5e3
GAMMA4 = 2 * np.pi * 3e3
GAMMA_34 = (GAMMA3 + GAMMA4) / 2

# Zeeman transitions between r and p sublevels
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
    """
    Polarization projection coefficient alpha_q(theta).
    Maps RF polarization angle to spherical-basis component weight:
      q=+1 (sigma+), q=-1 (sigma-), q=0 (pi).
    """
    if q == +1:
        return -1 / np.sqrt(2) * np.sin(theta)
    if q == -1:
        return 1 / np.sqrt(2) * np.sin(theta)
    if q == 0:
        return np.cos(theta)
    return 0.0


def omega_rf_components(theta_rf, theta_b, e0, b_field, omega_rf=OMEGA_RF_DEFAULT):
    """
    Build per-transition RF coupling data for the Zeeman-resolved model.
    Returns: list of (Omega_q, Delta_q, mJp, mJ)
    """
    components = []
    for tr in TRANSITIONS:
        # Zeeman shift projected along bias-field direction.
        delta_z = (G_P * tr["mJ"] - G_S * tr["mJp"]) * MU_B * b_field * np.cos(theta_b) / HBAR
        # Effective detuning for this path.
        delta_q = omega_rf - (OMEGA_0 + delta_z)
        # RF Rabi amplitude for this path.
        omega_q = (e0 / HBAR) * abs(alpha_pol(theta_rf, tr["q"]) * tr["phase"] * tr["CG"] * D_REDUCED)
        components.append((omega_q, delta_q, tr["mJp"], tr["mJ"]))
    return components


def h_system_6level(theta_rf, theta_b, e0, b_field, omega_rf=OMEGA_RF_DEFAULT):
    """
    Construct the full 6-level Hamiltonian:
      H = H_probe + H_coupling + H_rf + H_detuning
    """
    # Probe drives g <-> e
    h_p = (OMEGA_P / 2) * (SIG_GE + SIG_EG)
    # Coupling laser drives e <-> r+/-
    h_c = (OMEGA_C / 2) * (CG_CP_P * (SIG_ERP + SIG_RPE) + CG_CP_M * (SIG_ERM + SIG_RME))

    # RF term across Zeeman sublevels
    h_rf = Qobj(np.zeros((6, 6), dtype=complex))
    for omega_q, _, m_jp, m_j in omega_rf_components(theta_rf, theta_b, e0, b_field, omega_rf):
        idx_r = "r+" if m_jp == +0.5 else "r-"
        idx_p = "p+" if m_j == +0.5 else "p-"
        op_rp = STATES[idx_r] * STATES[idx_p].dag()
        h_rf += (omega_q / 2) * (op_rp + op_rp.dag())

    # Detuning shifts
    h_det = DELTA_P * PROJ["e"] + DELTA_C * (PROJ["r+"] + PROJ["r-"])
    for _, delta_q, _, m_j in omega_rf_components(theta_rf, theta_b, e0, b_field, omega_rf):
        key = "p+" if m_j == +0.5 else "p-"
        h_det += delta_q * PROJ[key]

    return h_p + h_c + h_rf + h_det


def collapse_operators_6level():
    """
    Dissipative channels for the 6-level Lindblad master equation:
    population decay (gamma2, gamma3, gamma4) and pure dephasing on p sublevels.
    """
    return [
        np.sqrt(GAMMA2) * SIG_GE,
        np.sqrt(GAMMA3) * (SIG_RPE + SIG_RME),
        np.sqrt(GAMMA4) * (STATES["r+"] * STATES["p+"].dag() + STATES["r-"] * STATES["p-"].dag()),
        np.sqrt(GAMMA_34) * PROJ["p+"],
        np.sqrt(GAMMA_34) * PROJ["p-"],
    ]
