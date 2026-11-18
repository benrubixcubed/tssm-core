from .dynamics import (
    TSSMParameters,
    compute_collapse_time,
    compute_asymptotic_peak,
    minimum_rest_duration,
    system_temperature,
    softmax_doxastic_policy,
    simulate_tssm,
)

try:
    from .agent import TSSMActiveInferenceAgent
except ImportError:
    TSSMActiveInferenceAgent = None

__all__ = [
    "TSSMParameters",
    "compute_collapse_time",
    "compute_asymptotic_peak",
    "minimum_rest_duration",
    "system_temperature",
    "softmax_doxastic_policy",
    "simulate_tssm",
    "TSSMActiveInferenceAgent",
]
