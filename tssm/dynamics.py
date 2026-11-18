from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class TSSMParameters:
    """Dimensionless scaling parameters for the Thermodynamic State-Space Model."""
    C0: float = 0.20          # Resting baseline executive cost
    C_crit: float = 1.00      # Critical executive threshold for phase collapse
    C_rec: float = 0.45       # Lower hysteretic restoration threshold (C_rec < C_crit)
    delta: float = 0.35       # Homeostatic metabolic clearance rate
    lambda_exec: float = 1.0  # Frontoparietal metabolic scaling constant
    kappa: float = 0.50       # Energetic return weight of free-energy reduction
    tau_0: float = 1.00       # Baseline exploratory system temperature
    tau_min: float = 0.01     # Functional temperature floor at dogmatic collapse
    k_gain: float = 35.0      # Steepness gain of hysteretic switching curve


def compute_collapse_time(L_on: float, params: TSSMParameters) -> float:
    """
    Analytical time to breach C_crit under continuous unfalsifiable deliberation (Eq. 6).
    Returns np.inf if steady-state ceiling C_ss <= C_crit.
    """
    threshold_gap = params.delta * (params.C_crit - params.C0)
    if L_on <= threshold_gap:
        return np.inf
    return -(1.0 / params.delta) * np.log(1.0 - (threshold_gap / L_on))


def compute_asymptotic_peak(
    L_on: float,
    L_rest: float,
    T_on: float,
    T_off: float,
    params: TSSMParameters,
) -> float:
    """
    Asymptotic periodic peak processing cost across alternating active/rest cycles (Eq. 7).
    """
    d = params.delta
    num = (
        L_on * (1.0 - np.exp(-d * T_on))
        + L_rest * np.exp(-d * T_on) * (1.0 - np.exp(-d * T_off))
    )
    den = d * (1.0 - np.exp(-d * (T_on + T_off)))
    return params.C0 + (num / den)


def minimum_rest_duration(L_on: float, T_on: float, params: TSSMParameters) -> float:
    """
    Minimum restorative rest duration T_off required to prevent multi-day collapse
    under zero-rumination ataraxia (L_rest = 0) (Eq. 8).
    """
    d = params.delta
    ratio = L_on / (d * (params.C_crit - params.C0))
    inner = (1.0 - ratio * (1.0 - np.exp(-d * T_on))) / np.exp(-d * T_on)
    if inner <= 0.0:
        return np.inf
    return max(0.0, -(1.0 / d) * np.log(inner))


def system_temperature(C: float, sigma: int, params: TSSMParameters) -> float:
    """
    Bistable hysteretic system temperature tau(C, sigma) governed by Schmitt-trigger state (Eq. 9).
    """
    c_theta = params.C_crit if sigma == 0 else params.C_rec
    exponent = np.clip(params.k_gain * (C - c_theta), -100.0, 100.0)
    return params.tau_min + (params.tau_0 - params.tau_min) / (1.0 + np.exp(exponent))


def softmax_doxastic_policy(
    expected_free_energy: np.ndarray,
    log_priors: np.ndarray,
    tau: float,
    noise: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Doxastic posterior distribution q(H_i) under inverse policy precision gamma = 1 / tau (Eq. 11).
    """
    gamma = 1.0 / max(tau, 1e-6)
    eps = np.zeros_like(expected_free_energy) if noise is None else noise
    values = -expected_free_energy + log_priors + eps
    scaled = gamma * values
    scaled -= np.max(scaled)
    exp_vals = np.exp(scaled)
    return exp_vals / np.sum(exp_vals)


def simulate_tssm(
    t_max: float = 16.0,
    dt: float = 0.01,
    L_on: float = 0.42,
    L_rest: float = 0.0,
    T_on: float = 2.0,
    T_off: float = 2.0,
    continuous: bool = False,
    abs_dF_dt: float = 0.0,
    params: Optional[TSSMParameters] = None,
) -> Dict[str, np.ndarray]:
    """
    Simulates executive processing cost C(t), binary Schmitt-trigger latch sigma(t),
    and exploratory system temperature tau(t) over time (Eqs. 4, 9, 10).
    """
    p = params or TSSMParameters()
    t = np.arange(0.0, t_max + dt, dt)
    n_steps = len(t)

    C = np.zeros(n_steps)
    sigma = np.zeros(n_steps, dtype=int)
    tau = np.zeros(n_steps)
    load_trace = np.zeros(n_steps)

    C[0] = p.C0
    sigma[0] = 0
    tau[0] = system_temperature(C[0], sigma[0], p)

    cycle_period = T_on + T_off

    for i in range(1, n_steps):
        if continuous:
            current_load = L_on
        else:
            phase = t[i - 1] % cycle_period
            current_load = L_on if phase < T_on else L_rest

        load_trace[i - 1] = current_load

        dC_dt = current_load - p.kappa * abs_dF_dt - p.delta * (C[i - 1] - p.C0)
        C[i] = max(0.0, C[i - 1] + dC_dt * dt)

        if sigma[i - 1] == 0 and C[i] >= p.C_crit:
            sigma[i] = 1
        elif sigma[i - 1] == 1 and C[i] <= p.C_rec:
            sigma[i] = 0
        else:
            sigma[i] = sigma[i - 1]

        tau[i] = system_temperature(C[i], sigma[i], p)

    load_trace[-1] = load_trace[-2]

    return {
        "t": t,
        "C": C,
        "sigma": sigma,
        "tau": tau,
        "load": load_trace,
    }
