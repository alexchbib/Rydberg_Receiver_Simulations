import qutip as qt
import numpy as np
import matplotlib.pyplot as plt


# initial state: 100% probability of finding the atom in the ground |0> state
rho_init = qt.Qobj(
    np.array([[1, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]])
)



# Hamiltonian under RWA
def H_RWA(
    Omega_p,
    Omega_c,
    Omega_RF,
    Delta_p,
    Delta_c,
    Delta_RF,
    hbar=1.0
):
    #Returns the 4-level Hamiltonian:
    #H = (ħ/2) * matrix

    H = (hbar / 2) * qt.Qobj([
        [0,                   Omega_p,                          0,                              0],
        [Omega_p,             -2*Delta_p,                       Omega_c,                        0],
        [0,                   Omega_c,                          -2* (Delta_p + Delta_c),              Omega_RF],
        [0,                   0,                                Omega_RF,                             -2*(Delta_p + Delta_c + Delta_RF)]
    ])

    return H



# Linblad operator representation

def decay_operators(gamma1, gamma2, gamma3, gamma4):
    #Returns collapse operators for a 4-level cascade system
    c_ops = []

    # |2> → |1>
    if gamma2 > 0:
        c_ops.append(np.sqrt(gamma2) * qt.basis(4, 0) * qt.basis(4, 1).dag())

    # |3> → |2>
    if gamma3 > 0:
        c_ops.append(np.sqrt(gamma3) * qt.basis(4, 1) * qt.basis(4, 2).dag())

    # |4> → |3>
    if gamma4 > 0:
        c_ops.append(np.sqrt(gamma4) * qt.basis(4, 2) * qt.basis(4, 3).dag())

    return c_ops
'''
def c_ops_exact(gamma):
    """
    Exact Lindblad operators reproducing Eq. (10)
    gamma = [gamma1, gamma2, gamma3, gamma4]
    """
    c_ops = []
    for i, g in enumerate(gamma):
        if g > 0:
            proj = qt.basis(4, i) * qt.basis(4, i).dag()
            c_ops.append(np.sqrt(g) * proj)
    return c_ops
'''


# Theoretical ρ_21
def rho21_analytic(Omega_p, Omega_c, Omega_RF,
                   Delta_p, Delta_c, Delta_RF,
                   gamma):
    """
    gamma = [gamma1, gamma2, gamma3, gamma4]
    """

    gamma1, gamma2, gamma3, gamma4 = gamma

    g21 = 0.5 * (gamma2 + gamma1)
    g31 = 0.5 * (gamma3 + gamma1)
    g41 = 0.5 * (gamma4 + gamma1)

    d3 = (g41 - 1j * (Delta_p + Delta_c + Delta_RF))
    d2 = (g31 - 1j * (Delta_p + Delta_c)) + (np.abs(Omega_RF)**2 / 4.0) / d3
    d1 = (g21 - 1j * Delta_p) + (np.abs(Omega_c)**2  / 4.0) / d2

    return (-1j * (Omega_p / 2.0)) / d1


# parameter values
H = H_RWA(0.04, 0.67, 0.8067, 0, 17.5, 0)
c_ops = decay_operators(0, 5.2, 3.9, 0.17)
tlist = np.linspace(0, 100000, 50000)


# Run the solver
result = qt.mesolve(
    H,
    rho_init,
    tlist,
    c_ops,
    []
)

# steady state
#rho_ss = result.states[-1]
#print(rho_ss)

rho_ss = qt.steadystate(H, c_ops, method='direct', tol=1e-12)
print(rho_ss)

# obtained ρ_21
rho21_numeric = rho_ss[1, 0]   # ρ_21


rho21_a = rho21_analytic(0.04, 0.67, 0.8067, 0, 17.5, 0, [0, 5.2, 3.9, 0.17])

print("Analytic rho_21 =", rho21_a)
print("Simulation  rho_21 =", rho21_numeric)
print("Difference      =", rho21_numeric - rho21_a)



#################
# Sweep Δc in MHz 
Delta_c_vals = np.linspace(-30, 30, 601)   # MHz
rho21_num = []
rho21_th  = []

for Dc in Delta_c_vals:

    # Hamiltonian for this Δc
    H = H_RWA(
        Omega_p=0.04,
        Omega_c=0.67,
        Omega_RF=0.8067,
        Delta_p=0,
        Delta_c=Dc,
        Delta_RF=0
    )

    # Steady state (QuTiP)
    rho_ss = qt.steadystate(H, c_ops, method='direct', tol=1e-12)
    rho21_num.append(rho_ss[1, 0])

    # Analytic value
    rho21_th.append(
        rho21_analytic(
            0.04, 0.67, 0.8067,
            0, Dc, 0,
            [0, 5.2, 3.9, 0.17]
        )
    )

rho21_num = np.array(rho21_num)
rho21_th  = np.array(rho21_th)



# Plot
plt.figure(figsize=(7,5))

plt.plot(
    Delta_c_vals,
    np.imag(rho21_num),
    label="QuTiP steady state",
    linewidth=2
)

plt.plot(
    Delta_c_vals,
    np.imag(rho21_th),
    "--",
    label="Analytic ρ21",
    linewidth=2
)

plt.xlabel(r"$\Delta_c$ (MHz)")
plt.ylabel(r"$\mathrm{Im}(\rho_{21})$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()