# Supplement S4.3: the NegBin DPH law without the stage cap, at the Gamma setting of s5_nonexp.jl (mean 9 d, shape 13.5).
# For each number of stages r, the NegBin DPH (total trials, support w >= r, p = per-day progression probability) that
# best fits the recorded Gamma interval law in the large-sample limit: p maximises E[log NB(δ; r, p) | δ >= r] under
# the recorded law. The recorded law follows s5_nonexp.jl (infection time uniform within its day):
# P(δ = d) = G(d+1) − 2G(d) + G(d−1), P(δ = 1) = G(2) − G(1), G(x) = ∫_0^x F_Gamma. Intervals shorter than r have zero
# likelihood under an r-stage NegBin DPH; their share of the recorded law is reported. No data or fits are used.
# Output: out/s5_negbin_uncapped.md (markdown table). Runs in under a second.
import math, os

k, mu = 13.5, 9.0
th = mu / k; h = 0.001; A = 60
n = int((A + 2) / h)
pdf = lambda a: 0.0 if a == 0 else math.exp((k - 1) * math.log(a) - a / th - math.lgamma(k) - k * math.log(th))
F = [0.0] * (n + 1); G = [0.0] * (n + 1); prev = 0.0
for i in range(1, n + 1):                                   # trapezoid integration of the PDF (F) and of F (G)
    cur = pdf(i * h); F[i] = min(F[i - 1] + h * (prev + cur) / 2, 1.0); prev = cur
    G[i] = G[i - 1] + h * (F[i - 1] + F[i]) / 2
s = round(1 / h)
Gi = lambda d: 0.0 if d <= 0 else G[d * s]
pc = {d: (Gi(2) - Gi(1) if d == 1 else Gi(d + 1) - 2 * Gi(d) + Gi(d - 1)) for d in range(1, A + 1)}
m = sum(d * p for d, p in pc.items()); sd = math.sqrt(sum((d - m) ** 2 * p for d, p in pc.items()))
sd_true = math.sqrt(mu * th)
lnb = lambda w, r, p: math.lgamma(w) - math.lgamma(r) - math.lgamma(w - r + 1) + r * math.log(p) + (w - r) * math.log1p(-p)

L = [f"Recorded Gamma law: mean {m:.2f} d, SD {sd:.2f} d (SD of the continuous Gamma {sd_true:.2f} d); "
     f"share of recorded intervals of 3 d or less {sum(pc[d] for d in (1, 2, 3)):.1e}.", "",
     "| Stages $r$ | Share of recorded intervals shorter than $r$ | Mean (d) | SD (d) | SD bias (%) |",
     "|---|---|---|---|---|"]
for r in range(1, 11):
    mass = sum(p for d, p in pc.items() if d >= r)
    fobj = lambda p: sum(pc[d] * lnb(d, r, p) for d in pc if d >= r) / mass
    lo, hi = 1e-6, 1 - 1e-9
    for _ in range(200):                                    # golden-section search on p (the objective is concave)
        a = hi - (hi - lo) / 1.618; b = lo + (hi - lo) / 1.618
        lo, hi = (a, hi) if fobj(a) < fobj(b) else (lo, b)
    p = (lo + hi) / 2
    L.append(f"| {r} | {1 - mass:.1e} | {r / p:.2f} | {math.sqrt(r * (1 - p)) / p:.2f} | {100 * (math.sqrt(r * (1 - p)) / p / sd_true - 1):+.1f} |")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "s5_negbin_uncapped.md")
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
