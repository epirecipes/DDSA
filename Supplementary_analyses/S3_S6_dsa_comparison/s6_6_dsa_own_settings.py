# S6: the authors' DSA Python code (Di Lauro et al. 2022; github Zkeggia/DSA_refactor @ 26671d6), used unchanged, in
# the setting of their own synthetic study (Synthetic_data/Code/1_Generate_data.py and 2_likelihood_li.py): Sellke
# outbreaks with N = 10,000, I_0 = 50, per-contact rate 0.25/(N - 1), Gamma infectious period (mean 9 d, variance 6),
# T_f = 80 d, kept if more than 200 events; infection times only (l_I), parameters (rho, beta, mean, variance)
# fitted by maximum likelihood with their particle swarm (100 particles x 100 iterations) within their bounds
# [5e-4, 0.01, 2, 1] - [1e-2, 2, 20, 15]. For comparison with our re-implementation (s6_7_dsa_own_settings.jl).
#   data <seed>         generate the outbreak with their Sellke_algo and write its infection times (as their script)
#   fit <seed> [grid]   their l_I maximum-likelihood fit (grid points over [0, T_f]; their script uses 10,000)
#   ll <seed> [grid]    their l_I at 12 parameter points (their likelihood class; its PDE starts the initial infectives
#                       with ages uniform on 0-5 d, finDiffUpdate's default, since the initial_cond it is passed is not read)
#   at <seed> [grid]    their l_I at their optimum and at the optimum of our re-implementation (s6_7_dsa_own_mle.csv)
# Usage (from the supplementary folder): <ext>/.venv/bin/python -I S3_S6_dsa_comparison/s6_6_dsa_own_settings.py <mode> <seed> [grid]
#   EXT=<folder> holds DSA_refactor (default S3_S6_dsa_comparison/external)
import sys, os, json, time
D = os.path.dirname(os.path.abspath(__file__))
EXT = os.environ.get("EXT", os.path.join(D, "external"))
sys.path[:0] = [os.path.join(D, "external", "shim"), os.path.join(EXT, "DSA_refactor")]
import numpy as np, scipy.stats as stats
from Likelihood import log_likelihood_models
from Selkealgo import Sellke_algo

mode, seed = sys.argv[1], int(sys.argv[2]); grid = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
DATA = os.path.join(D, "external", "data"); OUT = os.path.join(D, "out")
N, I_0, T_f = 10000, 50, 80
f_ti = os.path.join(DATA, f"dsaown_seed{seed}_infection_times.txt")

def rec_haz(u, *p):                      # as in their 2_likelihood_li.py: parameters (mean, variance)
    a = float(p[0]) ** 2 / float(p[1]); scale = float(p[1]) / float(p[0]); tol = 1e-10
    return np.where(stats.gamma.cdf(u, a=a, scale=scale) > 1 - tol, 1 / scale - (a - 1) / u,
                    stats.gamma.pdf(u, a=a, scale=scale) / (1 - stats.gamma.cdf(u, a=a, scale=scale)))
def rec_distr(u, *p):
    a = float(p[0]) ** 2 / float(p[1]); scale = float(p[1]) / float(p[0])
    return stats.gamma.pdf(u, a=a, scale=scale)
def inf_distr(u, *p): return float(p[0]) * np.ones_like(u)

if mode == "data":                       # their 1_Generate_data.py and the data extraction of 2_likelihood_li.py
    np.random.seed(seed)
    beta, mean, variance = 0.25, 9, 6
    scale = variance / mean; a = mean ** 2 / variance
    tau = beta / (N - 1)
    T_recover = stats.gamma.rvs(a=a, scale=scale, size=N)
    while True:
        t, I, S, R, _, _, _ = Sellke_algo(tau, I_0, N, T_f, T_recover, showplot=False, return_full=True)
        if len(t) > 200: break
    t, I, S = np.array(t), np.array(I), np.array(S)
    idx = np.where(N - S > 0)
    tl, Il = t[idx], I[idx]
    index_I = np.diff(Il, prepend=[Il[0]])
    np.savetxt(f_ti, tl[np.where(index_I > 0)])
    print(json.dumps({"seed": seed, "K": int(np.sum(index_I > 0)), "S_end": int(S[-1])}))
    sys.exit()

ti = np.loadtxt(f_ti)
ll = log_likelihood_models(grid, hazard_inf=inf_distr, hazard_rec=rec_haz, rec_distr=rec_distr,
                           T=T_f, infect_times=ti, hazard_inf_par=1, rec_parms=2)
if mode == "fit":
    np.random.seed(seed)
    t0 = time.time()
    x, f = ll.minimize_likelihood(np.array([5e-4, 0.01, 2, 1]), np.array([1e-2, 2, 20, 15]))
    print(json.dumps({"seed": seed, "grid": grid, "time_s": time.time() - t0, "rho": x[0], "beta": x[1], "mean": x[2],
                      "variance": x[3], "nll": float(f)}))
elif mode == "at":                       # our optimum is on our scale: back to theirs, rho_PDE = rho/(1 - rho), beta_PDE = beta/(1 + rho_PDE)
    import csv
    fit = [json.loads(l) for l in open(os.path.join(OUT, "s6_6_dsa_own_fits.jsonl")) if json.loads(l)["seed"] == seed][0]
    r = [x for x in csv.DictReader(open(os.path.join(OUT, "s6_7_dsa_own_mle.csv"))) if int(x["seed"]) == seed][0]
    rp = float(r["ours_rho"]) / (1 - float(r["ours_rho"]))
    xo = np.array([rp, float(r["ours_R0"]) / float(r["ours_mean"]) / (1 + rp), float(r["ours_mean"]), float(r["ours_var"])])
    xp = np.array([fit["rho"], fit["beta"], fit["mean"], fit["variance"]])
    print(json.dumps({"seed": seed, "grid": grid, "llI_python_at_python": float(-ll.evaluate_likelihood(xp)),
                      "llI_python_at_ours": float(-ll.evaluate_likelihood(xo)), "llI_ours_at_ours": float(r["ours_llI_at_ours"])}))
elif mode == "ll":
    rng = np.random.default_rng(11)
    pts = [np.array([50 / 9950, 0.25 * 9950 / 9999, 9.0, 6.0]) * np.exp(0.1 * rng.standard_normal(4)) for _ in range(12)]
    rows = []
    for x in pts:
        pub = -ll.evaluate_likelihood(x)
        rows.append({"seed": seed, "grid": grid, "rho": x[0], "beta": x[1], "mean": x[2], "variance": x[3],
                     "llI_published": float(pub)})
    with open(os.path.join(OUT, f"s6_6_dsa_own_ll_seed{seed}_grid{grid}.json"), "w") as io: json.dump(rows, io)
    print(json.dumps(rows[0]))
