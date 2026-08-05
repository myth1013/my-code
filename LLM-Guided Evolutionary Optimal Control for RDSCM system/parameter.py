import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True) 
class RDParams:
    # ---------------------- Spatial & Temporal Discretization Settings ----------------------
    X: float = 2.0  # Total length of 1D spatial domain Ω: x ∈ [0, X]
    T: float = 3.0  # Total simulation time horizon T ∈ (0, +∞)
    dx: float = 0.01  # Spatial step for finite difference spatial discretization
    dt: float = 5e-5  # Time step for fourth-order Runge-Kutta temporal discretization

    # ---------------------- RDSCM Epidemic & Physical Coefficients ----------------------
    Du: float = 0.001  # Diffusion coefficient for UWSN nodes
    Da: float = 0.001  # Diffusion coefficient for AUV nodes
    LA1: float = 0.08  # Birth rate of new susceptible UWSN nodes (Π_U in PDE system)
    LA2: float = 0.02  # Birth rate of new susceptible AUV nodes (Π_A in PDE system)
    alpha1: float = 0.4  # Recovery rate: Infected UWSN I_U → Recovered UWSN R_U (α₁)
    alpha2: float = 0.6  # Immunity loss rate: Recovered UWSN R_U → Susceptible UWSN S_U (α₂)
    beta1: float = 0.2  # Recovery rate: Infected AUV I_A → Recovered AUV R_A (β₁)
    beta2: float = 0.5  # Immunity loss rate: Recovered AUV R_A → Susceptible AUV S_A (β₂)
    d1: float = 0.005  # Natural damage/energy depletion mortality rate for all UWSN nodes (d₁)
    d2: float = 0.005  # Natural damage/energy depletion mortality rate for all AUV nodes (d₂)
    gamma1: float = 0.1  # Energy depletion rate: Normal UWSN nodes → Low-energy UWSN states (γ₁)
    gamma3: float = 0.3  # Wake-up rate: Sleep susceptible Ŝ_U → Normal susceptible S_U (γ₃)
    gamma4: float = 0.3  # Sleep switch rate: Normal susceptible S_U → Sleep susceptible Ŝ_U (γ₂ in paper)

    # ---------------------- Running Cost Weight Coefficients A ----------------------
    A1: float = 1.0  # Weight for UWSN infected node tracking loss in running cost
    A2: float = 1.0  # Weight for AUV infected node tracking loss in running cost
    A3: float = 1.0  # Weight for low-energy susceptible UWSN tracking loss in running cost
    A4: float = 1.0  # Weight for low-energy infected UWSN tracking loss in running cost
    A5: float = 1.0  # Weight for low-energy recovered UWSN tracking loss in running cost

    # ---------------------- Terminal Cost Weight Coefficients B ----------------------
    B1: float = 1.0  # Terminal loss weight for infected UWSN nodes
    B2: float = 1.0  # Terminal loss weight for infected AUV nodes
    B3: float = 1.0  # Terminal loss weight for low-energy susceptible UWSN nodes
    B4: float = 1.0  # Terminal loss weight for low-energy infected UWSN nodes
    B5: float = 1.0  # Terminal loss weight for low-energy recovered UWSN nodes

    # ---------------------- Control Cost Weight Coefficients C ----------------------
    C1: float = 0.5  # Cost weight of immune control for UWSN (u₁)
    C2: float = 0.3  # Cost weight of immune control for AUV (u₂)
    C3: float = 0.5  # Cost weight of wireless charging control (u₃)
    C4: float = 1.8  # Cost weight of intensified treatment for UWSN (u₄)
    C5: float = 5.5  # Cost weight of intensified treatment for AUV (u₅)
    C6: float = 2.1  # Cost weight of sleep isolation control (u₆)

    # ---------------------- Desired Steady-State Target Values ----------------------
    Iud: float = 1.2  # Target ideal density of infected UWSN nodes I_Uᵈ
    Iad: float = 1.2  # Target ideal density of infected AUV nodes I_Aᵈ
    LSud: float = 0.5  # Target ideal density of low-energy susceptible UWSN LS_Uᵈ
    LIud: float = 0.2  # Target ideal density of low-energy infected UWSN LI_Uᵈ
    LRud: float = 0.8  # Target ideal density of low-energy recovered UWSN LR_Uᵈ

    # ---------------------- Upper Bound of Each Admissible Control Variable ----------------------
    u_upper: tuple = (1.0, 1.0, 1.0, 0.5, 0.5, 0.5)  # Maximum feasible intensity for [u1, u2, u3, u4, u5, u6]

    # ---------------------- Derived Grid Calculation Methods ----------------------
    def get_M(self) -> int:
        """Calculate total spatial grid points M = X / dx"""
        return int(round(self.X / self.dx))

    def get_N(self) -> int:
        """Calculate total time iteration steps N = T / dt"""
        return int(round(self.T / self.dt))


# Global singleton parameter instance for whole simulation project
params_global = RDParams()


# ---------------------- Spatially Heterogeneous Infection Rate Functions (K(x) in Paper Sec. III-C) ----------------------
def K1_func(x: np.ndarray) -> np.ndarray:
    # Intra-UWSN space-varying malware transmission rate K₁(x)
    base = 0.0515
    return base + base * 0.4 * np.cos(x + np.pi)


def K2_func(x: np.ndarray) -> np.ndarray:
    # Intra-AUV space-varying malware transmission rate K₂(x)
    base = 0.061
    return base + base * 0.5 * np.cos(x + np.pi)


def K12_func(x: np.ndarray) -> np.ndarray:
    # Cross infection rate: UWSN → AUV K₁₂(x)
    base = 0.0034
    return base + base * 0.3 * np.cos(x + np.pi)


def K21_func(x: np.ndarray) -> np.ndarray:
    # Cross infection rate: AUV → UWSN K₂₁(x)
    base = 0.00064
    return base + base * 0.5 * np.cos(x + np.pi)


# ---------------------- Global Variable Export (Compatible with other simulation scripts) ----------------------
# Spatial & temporal discretization
X = params_global.X  # Total length of 1D spatial domain
T = params_global.T  # Total simulation time horizon
dx = params_global.dx  # Spatial discretization step
dt = params_global.dt  # Time discretization step
M = params_global.get_M()  # Total spatial grid quantity
N = params_global.get_N()  # Total time iteration steps

# Diffusion & birth coefficients
Du = params_global.Du  # Diffusion coefficient for UWSN nodes
Da = params_global.Da  # Diffusion coefficient for AUV nodes
LA1 = params_global.LA1  # Birth rate of susceptible UWSN nodes
LA2 = params_global.LA2  # Birth rate of susceptible AUV nodes

# Recovery & immunity loss & mortality rates
alpha1 = params_global.alpha1  # Recovery rate of infected UWSN
alpha2 = params_global.alpha2  # Immunity loss rate of recovered UWSN
beta1 = params_global.beta1  # Recovery rate of infected AUV
beta2 = params_global.beta2  # Immunity loss rate of recovered AUV
d1 = params_global.d1  # Mortality rate of UWSN nodes
d2 = params_global.d2  # Mortality rate of AUV nodes

# Energy consumption & sleep-wake transition rates
gamma1 = params_global.gamma1  # Energy depletion rate of UWSN nodes
gamma3 = params_global.gamma3  # Wake-up rate of sleep susceptible UWSN
gamma4 = params_global.gamma4  # Sleep switch rate of normal susceptible UWSN

# Running cost weights A
A1 = params_global.A1  # Loss weight for infected UWSN in running cost
A2 = params_global.A2  # Loss weight for infected AUV in running cost
A3 = params_global.A3  # Loss weight for low-energy susceptible UWSN in running cost
A4 = params_global.A4  # Loss weight for low-energy infected UWSN in running cost
A5 = params_global.A5  # Loss weight for low-energy recovered UWSN in running cost

# Terminal cost weights B
B1 = params_global.B1  # Terminal loss weight for infected UWSN
B2 = params_global.B2  # Terminal loss weight for infected AUV
B3 = params_global.B3  # Terminal loss weight for low-energy susceptible UWSN
B4 = params_global.B4  # Terminal loss weight for low-energy infected UWSN
B5 = params_global.B5  # Terminal loss weight for low-energy recovered UWSN

# Control cost weights C
C1 = params_global.C1  # Cost coefficient of UWSN immune control u1
C2 = params_global.C2  # Cost coefficient of AUV immune control u2
C3 = params_global.C3  # Cost coefficient of wireless charging control u3
C4 = params_global.C4  # Cost coefficient of UWSN intensified treatment u4
C5 = params_global.C5  # Cost coefficient of AUV intensified treatment u5
C6 = params_global.C6  # Cost coefficient of sleep isolation control u6

# Target ideal equilibrium states
Iud = params_global.Iud  # Expected equilibrium density of infected UWSN
Iad = params_global.Iad  # Expected equilibrium density of infected AUV
LSud = params_global.LSud  # Expected equilibrium density of low-energy susceptible UWSN
LIud = params_global.LIud  # Expected equilibrium density of low-energy infected UWSN
LRud = params_global.LRud  # Expected equilibrium density of low-energy recovered UWSN

# Upper bound of each control variable
u_max = list(params_global.u_upper)  # Maximum allowable intensity for six control inputs
