import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from tssm.dynamics import (
    TSSMParameters,
    compute_collapse_time,
    compute_asymptotic_peak,
    minimum_rest_duration,
    system_temperature,
    softmax_doxastic_policy,
    simulate_tssm,
)

st.set_page_config(page_title="TSSM Hysteresis Simulator", layout="wide")
st.title("Thermodynamic State-Space Model (TSSM) Simulator")
st.caption(
    "Interactive dynamical systems explorer for hysteretic dogmatic phase collapse, "
    "rumination ratcheting, and Epicurean ataraxia (Ruby, 2026)."
)

with st.sidebar:
    st.header("Executive & Metabolic Parameters")
    C0 = st.slider("Baseline Cost (C0)", 0.05, 0.40, 0.20, 0.01)
    C_crit = st.slider("Collapse Threshold (C_crit)", 0.70, 1.50, 1.00, 0.05)
    C_rec = st.slider("Restoration Threshold (C_rec)", 0.20, float(C_crit - 0.05), 0.45, 0.05)
    delta = st.slider("Metabolic Clearance Rate (delta)", 0.10, 0.80, 0.35, 0.05)

    st.header("Deliberation & Rumination Load")
    L_on = st.slider("Active Deliberation Load (L_on)", 0.10, 0.80, 0.42, 0.02)
    L_rest = st.slider("Tonic Rumination Load (L_rest)", 0.00, 0.40, 0.18, 0.02)
    T_on = st.slider("Active Bout Duration (T_on)", 0.5, 6.0, 2.0, 0.5)
    T_off = st.slider("Rest Bout Duration (T_off)", 0.5, 6.0, 2.0, 0.5)
    t_max = st.slider("Simulation Horizon (t_max)", 8.0, 32.0, 16.0, 2.0)

params = TSSMParameters(C0=C0, C_crit=C_crit, C_rec=C_rec, delta=delta)

sim_cont = simulate_tssm(t_max=t_max, L_on=L_on, continuous=True, params=params)
sim_rum = simulate_tssm(t_max=t_max, L_on=L_on, L_rest=L_rest, T_on=T_on, T_off=T_off, params=params)
sim_ata = simulate_tssm(t_max=t_max, L_on=L_on, L_rest=0.0, T_on=T_on, T_off=T_off, params=params)

t_col = compute_collapse_time(L_on, params)
c_peak_rum = compute_asymptotic_peak(L_on, L_rest, T_on, T_off, params)
c_peak_ata = compute_asymptotic_peak(L_on, 0.0, T_on, T_off, params)
t_off_min = minimum_rest_duration(L_on, T_on, params)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Continuous Collapse Time", f"{t_col:.2f}" if np.isfinite(t_col) else "Stable (inf)")
col2.metric("Asymptotic Peak (Rumination)", f"{c_peak_rum:.2f}")
col3.metric("Asymptotic Peak (Ataraxia)", f"{c_peak_ata:.2f}")
col4.metric("Min Rest for Ataraxia (T_off*)", f"{t_off_min:.2f}" if np.isfinite(t_off_min) else "Unbounded")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

c_grid = np.linspace(0.0, max(1.4, C_crit + 0.3), 400)
tau_upper = [system_temperature(c, 0, params) for c in c_grid]
tau_lower = [system_temperature(c, 1, params) for c in c_grid]

axes[0].plot(c_grid, tau_upper, label="Exploratory Branch (sigma=0)", color="#1f77b4", lw=2)
axes[0].plot(c_grid, tau_lower, label="Dogmatic Latch Branch (sigma=1)", color="#d62728", ls="--", lw=2)
axes[0].axvline(C_crit, color="#d62728", ls=":", alpha=0.7)
axes[0].axvline(C_rec, color="#2ca02c", ls=":", alpha=0.7)
axes[0].set_title("A. Bistable Hysteretic Phase Collapse")
axes[0].set_xlabel("Executive Processing Cost C(t)")
axes[0].set_ylabel("System Temperature tau")
axes[0].legend(fontsize=8)

axes[1].plot(sim_cont["t"], sim_cont["C"], label="Continuous Speculation", color="#d62728", lw=1.8)
axes[1].plot(sim_rum["t"], sim_rum["C"], label=f"Anxious Rest (L_rest={L_rest:.2f})", color="#ff7f0e", lw=1.8)
axes[1].plot(sim_ata["t"], sim_ata["C"], label="Ataraxia Rest (L_rest=0)", color="#2ca02c", lw=1.8)
axes[1].axhline(C_crit, color="black", ls="--", lw=1, label="C_crit")
axes[1].axhline(C_rec, color="gray", ls=":", lw=1, label="C_rec")
axes[1].set_title("B. Multi-Cycle Cost & Rumination Ratcheting")
axes[1].set_xlabel("Deliberation / Rest Time (t)")
axes[1].set_ylabel("Executive Processing Cost C(t)")
axes[1].legend(fontsize=8)

axes[2].plot(sim_rum["C"], sim_rum["tau"], color="#9467bd", lw=2)
axes[2].scatter(sim_rum["C"][0], sim_rum["tau"][0], color="#2ca02c", zorder=5, label="Start")
axes[2].scatter(sim_rum["C"][-1], sim_rum["tau"][-1], color="#d62728", zorder=5, label="End State")
axes[2].set_title("C. Phase Portrait (Anxious Trajectory)")
axes[2].set_xlabel("Executive Processing Cost C(t)")
axes[2].set_ylabel("System Temperature tau(t)")
axes[2].legend(fontsize=8)

plt.tight_layout()
st.pyplot(fig)

st.subheader("Doxastic Policy Posterior Under Tied Free Energy (Eq. 11)")
prior_bias = st.slider("Log-Prior Asymmetry ln P0(H1) - ln P0(H2)", 0.00, 0.50, 0.05, 0.01)
G_tied = np.array([1.0, 1.0])
log_P0 = np.array([prior_bias, 0.0])

q_exploratory = softmax_doxastic_policy(G_tied, log_P0, tau=params.tau_0)
q_collapsed = softmax_doxastic_policy(G_tied, log_P0, tau=params.tau_min)

st.write(
    f"**Exploratory State (tau = {params.tau_0:.2f}):** "
    f"P(H1) = {q_exploratory[0]:.3f}, P(H2) = {q_exploratory[1]:.3f}"
)
st.write(
    f"**Dogmatic Collapse (tau = {params.tau_min:.2f}):** "
    f"P(H1) = {q_collapsed[0]:.3f}, P(H2) = {q_collapsed[1]:.3f}"
)
