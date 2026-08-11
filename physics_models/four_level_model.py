"""
Four-Level Rydberg Cascade Atomic Model.

Defines the Hamiltonian under the Rotating Wave Approximation (RWA)
and Lindblad collapse operators for the 4-level cascade:

A regular quantum system with zero noise and zero decay follows the standard Schrödinger Equation.
But real-world atoms interact with the environment (they leak photons and lose energy). A system with decay/noise is called an Open Quantum System.
Göran Lindblad wrote down the standard equation that models open quantum systems with decay. That equation is called the Lindblad Master Equation.
In QuTiP, whenever you want to solve an open system with decay, you give it the Hamiltonian H and the list of collapse/decay operators c_ops.

"""

import numpy as np
import qutip as qt


def H_RWA(Omega_p, Omega_c, Omega_RF, Delta_p, Delta_c, Delta_RF, hbar=1.0): #Hamiltonian
    """
    Construct the 4-level rotating-wave Hamiltonian:
      H = (hbar / 2) * Matrix

    Basis:
      |1> = ground state
      |2> = intermediate excited state
      |3> = lower Rydberg state
      |4> = upper Rydberg state
    """
    matrix = [
        [0.0, Omega_p, 0.0, 0.0],
        [Omega_p, -2.0 * Delta_p, Omega_c, 0.0],
        [0.0, Omega_c, -2.0 * (Delta_p + Delta_c), Omega_RF],
        [0.0, 0.0, Omega_RF, -2.0 * (Delta_p + Delta_c + Delta_RF)],
    ]
    return (hbar / 2.0) * qt.Qobj(matrix)
    #qt.Qobj is a class that is used to represent quantum objects in qutip. In this case, i used it to represent the Hamiltonian of the system.


def decay_operators(gamma1, gamma2, gamma3, gamma4):
    """Decay operators are used to represent the decay of the system from higher energy levels to lower energy levels. 
        Construct Lindblad collapse operators for the cascade decay chain:
        |2> -> |1>, |3> -> |2>, |4> -> |3>.
    """
    c_ops = []
    if gamma2 > 0: #gamma2 is the decay rate of the |2> excited state to |1> ground state
        c_ops.append(np.sqrt(gamma2) * qt.basis(4, 0) * qt.basis(4, 1).dag())  
        """|1><2| drops an atom from state |2> down to state |1>
            qt.basis(4, 0) is the ground state |1> of the 4-level system.  
            qt.basis(4, 1) is the intermediate excited state |2> of the 4-level system.  
            .dag() is the conjugate transpose of the state.
            np.sqrt(gamma2) is the decay rate of the intermediate excited state.
        """
        
    if gamma3 > 0: #gamma3 is the decay rate of the |3> excited state to |2> excited state
        c_ops.append(np.sqrt(gamma3) * qt.basis(4, 1) * qt.basis(4, 2).dag())
    
    if gamma4 > 0: #gamma4 is the decay rate of the |4> excited state to |3> excited state
        c_ops.append(np.sqrt(gamma4) * qt.basis(4, 2) * qt.basis(4, 3).dag())
    return c_ops


def rho21_analytic(Omega_p, Omega_c, Omega_RF, Delta_p, Delta_c, Delta_RF, gamma): 
    """

    Why rho_21 and why Imaginary (Im)?
    - rho_21 is the optical coherence between Ground (Level 1) and Excited (Level 2).
    - The Imaginary part, Im(rho_21), represents optical absorption of the probe laser.
    - In the physical lab, the photodetector directly measures this absorption signal (EIT spectrum).
    - We use rho21_analytic as an exact pen-and-paper benchmark to verify QuTiP's numerical solver.
    
    
    rho_21 is the off-diagonal element of the density matrix that represents the coherence between the |1> ground state and the |2> intermediate excited state.
     rho21_analytic is the analytic solution for rho_21. 
    QuTiP solves the master equation using general numerical matrix algebra (qt.steadystate).
    We did this so that by comparing QuTiP's numerical result rho_ss[1, 0] against this exact analytical formula rho21_analytic, we can prove our QuTiP code has zero numerical bugs.
    gamma = [gamma1, gamma2, gamma3, gamma4]
    """
    gamma1, gamma2, gamma3, gamma4 = gamma

    g21 = 0.5 * (gamma2 + gamma1)
    g31 = 0.5 * (gamma3 + gamma1)
    g41 = 0.5 * (gamma4 + gamma1)

    d3 = g41 - 1j * (Delta_p + Delta_c + Delta_RF)
    d2 = (g31 - 1j * (Delta_p + Delta_c)) + (np.abs(Omega_RF) ** 2 / 4.0) / d3
    d1 = (g21 - 1j * Delta_p) + (np.abs(Omega_c) ** 2 / 4.0) / d2

    return (-1j * (Omega_p / 2.0)) / d1

"""
                  -i * (Omega_p / 2)
rho_21 = -------------------------------------------
                              (Omega_c / 2)^2
         d1 + --------------------------------------
                                    (Omega_RF / 2)^2
              d2 + ---------------------------------
                                  d3
where: 

d1 = (g21 - i * Delta_p) + (Omega_c / 2)^2 / d2
d2 = (g31 - i * (Delta_p + Delta_c)) + (Omega_RF / 2)^2 / d3
d3 = (g41 - i * (Delta_p + Delta_c + Delta_RF))
g21 = (gamma2 + gamma1) / 2
g31 = (gamma3 + gamma1) / 2
g41 = (gamma4 + gamma1) / 2
"""