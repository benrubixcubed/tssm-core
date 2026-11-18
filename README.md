# tssm-core: Thermodynamic State-Space Model (TSSM)

Open-source computational implementation of the **Thermodynamic State-Space Model (TSSM)** for active inference, neuroeconomics, and hysteretic belief collapse.

## Installation

```bash
pip install -e ".[all]"
```

## Quickstart

```python
from tssm import TSSMParameters, simulate_tssm, compute_collapse_time

params = TSSMParameters(C0=0.20, C_crit=1.00, C_rec=0.45, delta=0.35)
t_collapse = compute_collapse_time(L_on=0.42, params=params)
trajectory = simulate_tssm(t_max=16.0, L_on=0.42, L_rest=0.18, params=params)
```

## Interactive Simulator

```bash
streamlit run app.py
```

## Citations

- Ruby, B. (2026). *Thermodynamic Boundaries of Metaphysics: Resolving Epicurean Theology and Simulation Teleology through the Thermodynamic State-Space Model*. PhilArchive. https://philarchive.org/archive/RUBTBO-4
- Ruby, B. (2026). *A Hierarchical Thermodynamic Taxonomy of Doxastic States: An Active Inference Model, Neuro-Computational Opportunity Cost Architecture, and EEG-fMRI Fusion Paradigm*. Zenodo. https://doi.org/10.5281/zenodo.23677336
