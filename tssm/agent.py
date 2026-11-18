from typing import Dict, List, Optional
import numpy as np
from pymdp.agent import Agent
from .dynamics import TSSMParameters, system_temperature


class TSSMActiveInferenceAgent:
    """
    Wraps a standard pymdp Active Inference Agent with Thermodynamic State-Space Model (TSSM)
    metabolic cost tracking and bistable hysteretic policy-precision collapse.
    """

    def __init__(
        self,
        agent: Agent,
        params: Optional[TSSMParameters] = None,
        alpha_dacc: float = 0.85,
        dt: float = 1.0,
    ):
        self.agent = agent
        self.params = params or TSSMParameters()
        self.alpha_dacc = alpha_dacc
        self.dt = dt

        self.C: float = self.params.C0
        self.sigma: int = 0
        self.tau: float = system_temperature(self.C, self.sigma, self.params)
        self.agent.gamma = 1.0 / self.tau
        self.prev_free_energy: Optional[float] = None

        self.history: Dict[str, List[float]] = {
            "C": [],
            "sigma": [],
            "tau": [],
            "gamma": [],
            "policy_entropy": [],
            "abs_dF": [],
        }

    @staticmethod
    def _shannon_entropy(probs: np.ndarray) -> float:
        p = np.clip(probs, 1e-12, 1.0)
        return float(-np.sum(p * np.log(p)))

    def _compute_variational_free_energy(self, obs: List[int]) -> float:
        qs = self.agent.qs
        prior = self.agent.D
        kl_div = 0.0
        for f_idx, q_f in enumerate(qs):
            p_f = np.clip(prior[f_idx], 1e-12, 1.0)
            q_safe = np.clip(q_f, 1e-12, 1.0)
            kl_div += float(np.sum(q_safe * (np.log(q_safe) - np.log(p_f))))

        accuracy = 0.0
        for m_idx, o_val in enumerate(obs):
            likelihood_slice = self.agent.A[m_idx][o_val]
            expected_lik = likelihood_slice
            for f_idx, q_f in enumerate(qs):
                expected_lik = np.tensordot(expected_lik, q_f, axes=(0, 0))
            accuracy += float(np.log(np.clip(expected_lik, 1e-12, 1.0)))

        return kl_div - accuracy

    def step(self, obs: List[int], is_resting: bool = False, L_rest: float = 0.0) -> np.ndarray:
        self.agent.infer_states(obs)
        current_F = self._compute_variational_free_energy(obs)
        abs_dF = 0.0 if self.prev_free_energy is None else abs(current_F - self.prev_free_energy)
        self.prev_free_energy = current_F

        self.agent.gamma = 1.0 / max(self.tau, 1e-6)
        q_pi, _ = self.agent.infer_policies()
        h_policy = self._shannon_entropy(q_pi)

        if is_resting:
            active_load = L_rest
            effective_dF = 0.0
        else:
            active_load = self.params.lambda_exec * self.alpha_dacc * h_policy
            effective_dF = abs_dF / max(self.dt, 1e-6)

        dC_dt = (
            active_load
            - self.params.kappa * effective_dF
            - self.params.delta * (self.C - self.params.C0)
        )
        self.C = max(0.0, self.C + dC_dt * self.dt)

        if self.sigma == 0 and self.C >= self.params.C_crit:
            self.sigma = 1
        elif self.sigma == 1 and self.C <= self.params.C_rec:
            self.sigma = 0

        self.tau = system_temperature(self.C, self.sigma, self.params)
        self.agent.gamma = 1.0 / max(self.tau, 1e-6)

        self.history["C"].append(self.C)
        self.history["sigma"].append(self.sigma)
        self.history["tau"].append(self.tau)
        self.history["gamma"].append(self.agent.gamma)
        self.history["policy_entropy"].append(h_policy)
        self.history["abs_dF"].append(abs_dF)

        return self.agent.sample_action()
      
