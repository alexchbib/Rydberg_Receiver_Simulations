import math
from sympy import S
from sympy.physics.wigner import wigner_3j
import numpy as np
import matplotlib.pyplot as plt


def _halfint(x: float):
    return S(int(round(2*x))) / 2



# ---- Eq.18 ----

def analytical_Zeeman_resolved_rho_21(Omega_p, Omega_c,
                                     delta_p, delta_c,
                                     gamma1, gamma2, gamma3,
                                     sum_E1):
    d2 = delta_c + 1j * ((gamma1 + gamma3)/2) - sum_E1
    d1 = delta_p + 1j * ((gamma1 + gamma2)/2) - (((Omega_c/2)**2) / d2)
    return (1j * (Omega_p/2) / d1)


# ---- Eq.17 ----

def delta_Z_E1(mu_B, B_bias, theta_bias,
               g_P, m_J, g_S, m_p_J,
               h_bar):
    return (((mu_B * B_bias)/h_bar) * (g_P*m_J - g_S*m_p_J) * math.cos(theta_bias))


# ---- Eq.16 ----

def delta_E1(w_RF, w_0, delta_Z_E1):
    return w_RF - (w_0 + delta_Z_E1)


# ---- Eq.15 ----

def Omega_E1(q: int, theta_RF: float, m_J: float, m_p_J: float,
             E0: float, d_reduced_SI: float, h_bar):
    if q not in (-1, 0, 1):
        raise ValueError("q must be -1, 0, +1")

    if q == 0:
        alpha = math.cos(theta_RF)
    else:
        alpha = (-q) * math.sin(theta_RF) / math.sqrt(2)

    mJ  = _halfint(m_J)
    mJp = _halfint(m_p_J)

    phase  = (-1) ** int(S(1)/2 - mJ)
    threej = float(wigner_3j(S(1)/2, 1, S(1)/2, -mJ, q, mJp))

    return (E0 / h_bar) * abs(alpha * phase * threej * d_reduced_SI) / (2 * math.pi * 1e6)


# ---- Eq.14 ---- per path

def self_energy_per_transition_path(theta_RF,
                                    q, m_J, mp_J,
                                    E0, d_reduced_SI, h_bar,
                                    w_RF, w_0,
                                    mu_B, B_bias, theta_bias, g_P, g_S,
                                    gamma_RF):
    rabi_freq_E1 = Omega_E1(q, theta_RF, m_J, mp_J, E0, d_reduced_SI, h_bar)
    zeeman_shift_E1 = delta_Z_E1(mu_B, B_bias, theta_bias, g_P, m_J, g_S, mp_J, 1)
    detuning = delta_E1(w_RF, w_0, zeeman_shift_E1)

    return ((abs(rabi_freq_E1))**2) / (detuning + 1j * gamma_RF)


# ---- Eq.14 ---- 

def self_energy(theta_RF,
                E0, d_reduced_SI, h_bar,
                w_RF, w_0,
                mu_B, B_bias, theta_bias, g_P, g_S,
                gamma_RF):
    tp1 = self_energy_per_transition_path(theta_RF, -1, -0.5, 0.5, E0, d_reduced_SI, h_bar, w_RF, w_0, mu_B, B_bias, theta_bias, g_P, g_S, gamma_RF)
    tp2 = self_energy_per_transition_path(theta_RF, 1, +0.5, -0.5, E0, d_reduced_SI, h_bar, w_RF, w_0, mu_B, B_bias, theta_bias, g_P, g_S, gamma_RF)
    tp3 = self_energy_per_transition_path(theta_RF, 0, +0.5, +0.5, E0, d_reduced_SI, h_bar, w_RF, w_0, mu_B, B_bias, theta_bias, g_P, g_S, gamma_RF)
    tp4 = self_energy_per_transition_path(theta_RF, 0, -0.5, -0.5, E0, d_reduced_SI, h_bar, w_RF, w_0, mu_B, B_bias, theta_bias, g_P, g_S, gamma_RF)
    return tp1 + tp2 + tp3 + tp4


#-------------------------------------------------------------------------------------------------------------------------------------------------------

# ---- Parameters ----

# Reduced Plank Constant
h_bar = 1.054571817e-34

# Rabi Frequencies
Omega_p = 6         # MHz
Omega_c = 0.67      # MHz
omega_0 = 6900      # MHz
omega_RF = omega_0 + 0.8        # MHz

# Inverse Lifetimes
gamma_1 = 0         # MHz
gamma_2 = 5.2       # MHz
gamma_3 = 3.9       # MHz
gamma_4 = 0.17      # MHz
gamma_RF = 10e-3    # MHz

# Detunings
delta_p = 0
delta_RF = 0

# Reduced Dipole Moment
a0 = 5.2e-11        # m ; Bohr Radius
e = 1.6e-19         # C ; Elementary Charge
d = -1443.46 * e * a0

# Bias Field Parameters
B_bias = 0.002      # T
theta_bias = 0
mu_B = 13996        # MHz / Tesla ; Bohr Magneton


# Electric Field Amplitude
e0 = 1              # V/m

# Lande Factors
g_S = 2.0
g_P = 0.67

# theta_RF
theta_RF = 30 * np.pi / 180


#-------------------------------------------------------------------------------------------------------------------------------------------------------

delta_c = np.linspace(-10, 30, 4000)


tp1 = self_energy_per_transition_path(theta_RF, -1, -0.5, 0.5, e0, d, h_bar, omega_RF, omega_0, mu_B, B_bias, theta_bias, g_P, g_S, gamma_RF)
tp2 = self_energy_per_transition_path(theta_RF, 1, 0.5, -0.5, e0, d, h_bar, omega_RF, omega_0, mu_B, B_bias, theta_bias, g_P, g_S, gamma_RF)
tp3 = self_energy_per_transition_path(theta_RF, 0, +0.5, +0.5, e0, d, h_bar, omega_RF, omega_0, mu_B, B_bias, theta_bias, g_P, g_S, gamma_RF)
tp4 = self_energy_per_transition_path(theta_RF, 0, -0.5, -0.5, e0, d, h_bar, omega_RF, omega_0, mu_B, B_bias, theta_bias, g_P, g_S, gamma_RF)

s_energy = self_energy(theta_RF, e0, d, h_bar, omega_RF, omega_0, mu_B, B_bias, theta_bias, g_P, g_S, gamma_RF)

print(f"tp1 {tp1} \ntp2 {tp2} \ntotal {s_energy}")
rho_0 = analytical_Zeeman_resolved_rho_21(Omega_p, Omega_c, delta_p, delta_c, gamma_1, gamma_2, gamma_3, 0)
rho_1 = analytical_Zeeman_resolved_rho_21(Omega_p, Omega_c, delta_p, delta_c, gamma_1, gamma_2, gamma_3, (tp1))
rho_2 = analytical_Zeeman_resolved_rho_21(Omega_p, Omega_c, delta_p, delta_c, gamma_1, gamma_2, gamma_3, (tp2))
rho_3 = analytical_Zeeman_resolved_rho_21(Omega_p, Omega_c, delta_p, delta_c, gamma_1, gamma_2, gamma_3, (tp3))
rho_4 = analytical_Zeeman_resolved_rho_21(Omega_p, Omega_c, delta_p, delta_c, gamma_1, gamma_2, gamma_3, (tp4))
rho_t_1 = analytical_Zeeman_resolved_rho_21(Omega_p, Omega_c, delta_p, delta_c, gamma_1, gamma_2, gamma_3, s_energy)
rho_t_2 = (np.imag(rho_0)) + (np.imag(rho_0)) + (np.imag(rho_0)) + (np.imag(rho_0))

for q in [-1,0,+1]:
    for mJp in [-0.5, +0.5]:
        mJ = mJp + q
        print(q, mJp, mJ, Omega_E1(q, theta_RF, mJ, mJp, e0, d, h_bar))


print(np.imag(rho_0))
print(np.imag(rho_1))
print(np.imag(rho_2))
print(np.imag(rho_3))


plt.figure()

plt.plot(delta_c, -np.imag(rho_1))
plt.plot(delta_c, -(np.imag(rho_2)))
plt.plot(delta_c, -(np.imag(rho_3)))
plt.plot(delta_c, -(np.imag(rho_4)))

plt.plot(delta_c, (np.imag(rho_t_1)))
plt.plot(delta_c, ((rho_t_2)))

plt.xlim(-10,30)
plt.xlabel(r'$\Delta_c/2\pi$ (MHz)')
plt.ylabel(r'Im($\rho_{21}$)')
plt.title(r'Zeeman-resolved EIT ($\theta_{RF}=30^\circ,\ \theta_{bias}=0^\circ$)')
plt.show()




delta_RF = np.linspace(-100, 100, 4000)   # MHz, RF detuning
rho_vs_RF = np.zeros_like(delta_RF, dtype=complex)

for i, dRF in enumerate(delta_RF):
    w_RF_i = omega_0 + dRF   # RF frequency in MHz

    # self-energy for this RF frequency
    s_energy_i = self_energy(
        theta_RF,
        e0, d, h_bar,
        w_RF_i, omega_0,
        mu_B, B_bias, theta_bias,
        g_P, g_S,
        gamma_RF
    )

    rho_vs_RF[i] = analytical_Zeeman_resolved_rho_21(
        Omega_p, Omega_c,
        delta_p,  # still 0
        0.0,      # Δc fixed at 0 for this scan
        gamma_1, gamma_2, gamma_3,
        s_energy_i
    )

plt.figure()
plt.plot(delta_RF, -np.imag(rho_vs_RF))
plt.xlabel(r'$\Delta_{\mathrm{RF}}/2\pi$ (MHz)')
plt.ylabel(r'Im($\rho_{21}$)')
plt.title(r'Zeeman-resolved EIT vs RF detuning')
plt.show()
