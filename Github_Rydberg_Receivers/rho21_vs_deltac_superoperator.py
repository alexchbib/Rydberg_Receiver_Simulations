# 4-level Rydberg receiver (QuTiP) SUPEROPERATOR

import numpy as np
import matplotlib.pyplot as plt

from qutip import (
    Qobj, basis, steadystate,
    spre, spost, operator_to_vector
)

two_pi = 2 * np.pi

def to_rad(mhz):
    """Convert MHz -> angular units (rad / microsecond). Consistent internal units."""
    return two_pi * mhz

# Hamiltonian 
def build_hamiltonian(omega_p, omega_c, omega_rf, delta_p, delta_c, delta_rf):
    """
    Basis ordering:
      |1> = 5S1/2      -> index 0
      |2> = 5P3/2      -> index 1
      |3> = n'S1/2     -> index 2
      |4> = nP1/2      -> index 3

    H = (1/2) * [[0, Ωp, 0, 0],
                 [Ωp, -2Δp, -Ωc, 0],
                 [0, Ωc, -2(Δp+Δc), ΩRF],
                 [0, 0, ΩRF, -2(Δp+Δc+ΔRF)]]
    """
    m = np.array([
        [0,                 omega_p,                   0,                    0],
        [omega_p, -2*delta_p,                  omega_c,             0],
        [0,                 omega_c,         -2*(delta_p + delta_c), omega_rf],
        [0,                  0,                        omega_rf,   -2*(delta_p + delta_c + delta_rf)]
    ], dtype=complex)

    return 0.5 * Qobj(m)

# Dissipator as a superoperator 

def build_D_super(gamma1, gamma2, gamma3, gamma4, gamma_rf_deph=0.0):
    """
    Build 16x16 superoperator D such that vec(L(ρ)) = D vec(ρ),
    where L(ρ) matches the equation:
    L = [[γ2 ρ22,   -γ12 ρ12,           -γ13 ρ13,           -γ14 ρ14],
        [-γ21 ρ21,  γ3 ρ33 - γ2 ρ22,    -γ23 ρ23,           -γ24 ρ24],
        [-γ31 ρ31,  -γ32 ρ32,           γ4 ρ44 - γ3 ρ33,    -γ34 ρ34],
        [-γ41 ρ41,   -γ42 ρ42,           -γ43 ρ43,           -γ4 ρ44]]

    Diagonals (population flow):
      L_11 = +γ_2 ρ_22
      L_22 = +γ_3 ρ_33 - γ2 ρ_22
      L_33 = +γ_4 ρ_44 - γ3 ρ_33
      L_44 = -γ_4 ρ_44

    Off-diagonals:
      L_ij = -γ_ij ρ_ij   for i≠j,  γ_ij = (γ_i + γ_j)/2

    Optional: RF-transition extra dephasing gamma_rf_deph applied ONLY to coherences ρ_34 and ρ_43.
      (|3><4| and |4><3|) i.e. indices (2,3) and (3,2).
    """
    N = 4
    gam = np.array([gamma1, gamma2, gamma3, gamma4], dtype=float)

    def gamma_ij(i, j):
        return 0.5 * (gam[i] + gam[j])

    def L_action(rho):
        r = rho.full()
        out = np.zeros((N, N), dtype=complex)

        # --- diagonals (cascade) ---
        out[0, 0] = gam[1] * r[1, 1]
        out[1, 1] = gam[2] * r[2, 2] - gam[1] * r[1, 1]
        out[2, 2] = gam[3] * r[3, 3] - gam[2] * r[2, 2]
        out[3, 3] = -gam[3] * r[3, 3]

        # --- off-diagonals (phenomenological damping) ---
        for i in range(N):
            for j in range(N):
                if i != j:
                    out[i, j] = -gamma_ij(i, j) * r[i, j]

        # --- optional extra RF dephasing ONLY on the (3<->4) coherence ---
        if gamma_rf_deph > 0:
            out[2, 3] += -gamma_rf_deph * r[2, 3]
            out[3, 2] += -gamma_rf_deph * r[3, 2]

        # If you wanted to literally hard-code the Eq.(10) typo, you could do it here,
        # but it doesn't make sense physically. We'll ignore that and keep the clean rule.

        return Qobj(out, dims=rho.dims)

    # Build superoperator by its action on the operator basis E_ij = |i><j|
    D = np.zeros((N * N, N * N), dtype=complex)
    for i in range(N):
        for j in range(N):
            Eij = basis(N, i) * basis(N, j).dag()  # |i><j|
            col = operator_to_vector(L_action(Eij)).full()  # vec(L(Eij))
            col_index = i + j * N  # column-stacking (QuTiP convention)
            D[:, col_index] = col[:, 0]

    return Qobj(D, dims=[[[N], [N]], [[N], [N]]])

def build_liouvillian(H, gamma1, gamma2, gamma3, gamma4, gamma_rf_deph=0.0):
    """
    Full Liouvillian superoperator:
      L = -i (spre(H) - spost(H)) + D
    """
    D = build_D_super(gamma1, gamma2, gamma3, gamma4, gamma_rf_deph=gamma_rf_deph)
    L = -1j * (spre(H) - spost(H)) + D
    return L

# Analytic rho21 (for comparison)
def rho21_analytic(omega_p, omega_c, omega_rf, delta_p, delta_c, delta_rf,
                  gamma1, gamma2, gamma3, gamma4,
                  sign_in_cf="+"):
    """
    Continued-fraction form (weak-probe / linear response).

    rho21 = -i(Ωp/2) / [ γ21 - iΔp  (sign) (|Ωc|/2)^2 / ( γ31 - i(Δp+Δc)  (sign) (|ΩRF|/2)^2 / ( γ41 - i(Δp+Δc+ΔRF) ) ) ]

    sign_in_cf:
      "+" usually matches the Hamiltonian sign convention used above.
      "-" corresponds to the alternative convention found in some derivations.
    """
    g21 = 0.5 * (gamma2 + gamma1)
    g31 = 0.5 * (gamma3 + gamma1)
    g41 = 0.5 * (gamma4 + gamma1)

    d3 = (g41 - 1j * (delta_p + delta_c + delta_rf))

    if sign_in_cf == "+":
        d2 = (g31 - 1j * (delta_p + delta_c)) + (np.abs(omega_rf)**2 / 4.0) / d3
        d1 = (g21 - 1j * delta_p)             + (np.abs(omega_c)**2  / 4.0) / d2
    else:
        d2 = (g31 - 1j * (delta_p + delta_c)) - (np.abs(omega_rf)**2 / 4.0) / d3
        d1 = (g21 - 1j * delta_p)             - (np.abs(omega_c)**2  / 4.0) / d2

    return (-1j * (omega_p / 2.0)) / d1

# Run example sweep: Δc
if __name__ == "__main__":

    # --- Parameters (from your text) ---
    omega_p  = to_rad(0.04)   # Ωp/2π = 0.04 MHz
    omega_c  = to_rad(0.67)   # Ωc/2π = 0.67 MHz

    # ΩRF amplitude is usually a tunable knob; choose something and tweak:
    omega_rf = to_rad(0.50)   # ΩRF/2π = 0.50 MHz (change to match paper fig)

    delta_p  = to_rad(0.0)    # Δp = 0
    delta_rf = to_rad(0.0)    # ΔRF = 0 (common in plots)

    # Decay rates (given as γ/2π in MHz)
    gamma1 = to_rad(0.0)      # ground decay neglected
    gamma2 = to_rad(5.2)
    gamma3 = to_rad(3.9)
    gamma4 = to_rad(0.17)

    # RF transition dephasing (10 kHz = 0.01 MHz). If you want EXACT Eq.(10), set 0.0.
    gamma_rf_deph = to_rad(0.0)   # to_rad(0.01) if you want to include it

    # Sweep Δc/(2π) in MHz
    dc_vals_mhz = np.linspace(-30, 30, 601)
    dc_vals = to_rad(dc_vals_mhz)

    rho21_num = np.zeros_like(dc_vals, dtype=complex)
    rho21_th  = np.zeros_like(dc_vals, dtype=complex)

    for k, dc in enumerate(dc_vals):
        H = build_hamiltonian(omega_p, omega_c, omega_rf, delta_p, dc, delta_rf)
        L = build_liouvillian(H, gamma1, gamma2, gamma3, gamma4, gamma_rf_deph=gamma_rf_deph)

        rho_ss = steadystate(L)           # steady state density matrix
        rho21_num[k] = rho_ss[1, 0]       # ρ21 = <2|ρ|1>

        # Optional analytic comparison (pick sign "+" or "-")
        rho21_th[k] = rho21_analytic(
            omega_p, omega_c, omega_rf,
            delta_p, dc, delta_rf,
            gamma1, gamma2, gamma3, gamma4,
            sign_in_cf="+"
        )

    # Plot absorption-like signal (many papers plot Im(rho21))
    plt.figure()
    plt.plot(dc_vals_mhz, np.imag(rho21_num), label="QuTiP Im(rho21)")
    plt.plot(dc_vals_mhz, np.imag(rho21_th),  "--", label="Analytic Im(rho21)")
    plt.xlabel("Δc / (2π) [MHz]")
    plt.ylabel("Im(ρ21)")
    plt.title("4-level Rydberg receiver: steady-state ρ21 vs Δc")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("max |rho21_num - rho21_analytic| =", np.max(np.abs(rho21_num - rho21_th)))
