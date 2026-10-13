# S6: the authors' DSA Python code (Di Lauro et al. 2022; github Zkeggia/DSA_refactor @ 26671d6), used unchanged via
# its log_likelihood_models class, on exported S5 datasets. Gamma infectious period, exponential contact interval
# (constant infection hazard β), parameters (ρ, β, shape, scale) fitted by maximum likelihood with their particle-swarm
# optimiser (pyswarms GlobalBestPSO, 100 particles × 100 iterations, c1 = c2 = 1.7, w = 0.6, as in their code), PDE grid
# 0.1 d. Scenario A: infection times (ℓ_I) + infectious periods (ℓ_R1); scenario B: infection times only. Their mean
# field is per susceptible, so β = β_DSA (1 + ρ_DSA) and R0 = β × shape × scale on the simulator's scale.
# Usage (from the supplementary folder): PYTHONPATH=<ext>/shim:<ext>/DSA_refactor <ext>/.venv/bin/python s6_3_dsa_python.py <rep> <A|B> [iters]
# (<ext> = S3_S6_dsa_comparison/external; get_authors_code.sh downloads their code there)
import sys, time, json, os
import numpy as np, scipy.stats as stats
from Likelihood import log_likelihood_models

rep, scen = int(sys.argv[1]), sys.argv[2]; iters = int(sys.argv[3]) if len(sys.argv) > 3 else 100
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "external", "data")
ti = np.sort(np.loadtxt(f"{D}/dsa5_rep{rep}_infection_times.txt"))
ages = np.loadtxt(f"{D}/dsa5_rep{rep}_infectious_periods.txt")
rt = np.loadtxt(f"{D}/dsa5_rep{rep}_recovery_times.txt")

def inf_haz(u, *p): return float(p[0]) * np.ones_like(u)
def rec_haz(u, *p):                     # as in their Examples/Pde_inference.py
    a, scale = p[0], p[1]; tol = 1e-10; x = np.zeros_like(u)
    bad = stats.gamma.cdf(u, a=a, scale=scale) > 1 - tol
    x[bad] = 1 / scale - (a - 1) / u[bad]
    x[~bad] = stats.gamma.pdf(u[~bad], a=a, scale=scale) / (1 - stats.gamma.cdf(u[~bad], a=a, scale=scale))
    return x
def rec_distr(u, *p):
    if len(p) == 1: p = p[0]
    return stats.gamma.pdf(u, a=p[0], scale=p[1])

T_f = float(max(rt.max(), ti.max()))
kw = dict(hazard_inf=inf_haz, hazard_rec=rec_haz, rec_distr=rec_distr, T=T_f, infect_times=ti,
          hazard_inf_par=1, rec_parms=2)
if scen == "A": kw["infectious_ages"] = ages
ll = log_likelihood_models(int(np.ceil(T_f / 0.1)), **kw)
lb = np.array([0.001, 0.05, 1.0, 0.1]); ub = np.array([0.02, 1.0, 30.0, 5.0])
t0 = time.time()
if iters == 0:                          # timing of single evaluations only
    x = np.array([50 / 9950, 0.25, 13.5, 9 / 13.5])
    for _ in range(3): v = ll.evaluate_likelihood(x)
    print(json.dumps({"rep": rep, "scen": scen, "eval_s": (time.time() - t0) / 3, "nll_truth": float(v)})); sys.exit()
np.random.seed(rep)
xopt, fopt = ll.minimize_likelihood(lb, ub, maxiter=iters, swarmsize=100)
el = time.time() - t0
# pyswarms returns (best cost, best position); their code names them (fopt, xopt) in that order
cost, pos = (xopt, fopt) if np.ndim(xopt) == 0 else (fopt, xopt)
rho, b, a, sc = pos; beta = b * (1 + rho); mu = a * sc
print(json.dumps({"rep": rep, "scen": scen, "time_s": el, "rho": rho, "beta": beta, "shape": a, "scale": sc,
                  "mu": mu, "sd": sc * np.sqrt(a), "R0": beta * mu, "nll": float(cost)}))
