"""
Theoretical & Literature Benchmarks (Chen et al. Replication).

Contains analytical continued fractions, Wigner-3j self-energy formulas,
and superoperator reference implementations for benchmark verification.
"""

import math
import numpy as np
from sympy import S
from sympy.physics.wigner import wigner_3j


def analytical_zeeman_resolved_rho21(omega_p, omega_c, delta_p, delta_c, gamma1, gamma2, gamma3, sum_e1):
    """Replication of Eq. 18 (Chen et al.)."""
    d2 = delta_c + 1j * ((gamma1 + gamma3) / 2) - sum_e1
    d1 = delta_p + 1j * ((gamma1 + gamma2) / 2) - (((omega_c / 2) ** 2) / d2)
    return 1j * (omega_p / 2) / d1


def self_energy_per_transition_path(theta_rf, q, m_j, mp_j, e0, d_reduced, hbar, w_rf, w_0, mu_b, b_bias, theta_bias, g_p, g_s, gamma_rf):
    """Compute transition self-energy for one path."""
    if q == 0:
        alpha = math.cos(theta_rf)
    else:
        alpha = (-q) * math.sin(theta_rf) / math.sqrt(2)

    m_j_sym = S(int(round(2 * m_j))) / 2
    mp_j_sym = S(int(round(2 * mp_j))) / 2
    phase = (-1) ** int(S(1) / 2 - m_j_sym)
    threej = float(wigner_3j(S(1) / 2, 1, S(1) / 2, -m_j_sym, q, mp_j_sym))

    rabi = (e0 / hbar) * abs(alpha * phase * threej * d_reduced) / (2 * math.pi * 1e6)
    zeeman_shift = ((mu_b * b_bias) / 1.0) * (g_p * m_j - g_s * mp_j) * math.cos(theta_bias)
    detuning = w_rf - (w_0 + zeeman_shift)

    return (abs(rabi) ** 2) / (detuning + 1j * gamma_rf)
