# Writes the LaTeX tables of supplement Sections S3–S7 from the result files, so the manuscript numbers stay in sync
# with the runs. Output: supp_tables/*.tex in this folder (\input by the supplement). Tables whose runs are
# not finished are written as a "pending" placeholder.
# Usage (from the supplementary folder): python3 -I make_supplement_tables.py
import csv, math, os, random, statistics as st, collections, json
SA = os.path.dirname(os.path.abspath(__file__))
D3 = os.path.join(SA, "S3_S6_dsa_comparison", "out"); OUT = os.path.join(SA, "supp_tables")
os.makedirs(OUT, exist_ok=True)
def load(p): return list(csv.DictReader(open(p))) if os.path.exists(p) else []
def f(x):
    try: return float(x)
    except: return float("nan")
def write(name, body): open(os.path.join(OUT, name), "w").write(body); print("wrote", name)
def pending(name, what): write(name, "\\begin{tabular}{c}\\emph{" + what + ": results pending.}\\end{tabular}\n")
def met(rows, p, sd="sd"):
    rows = [r for r in rows if not math.isnan(f(r[f"{p}_med"]))]
    if len(rows) < 2: return None
    t = [f(r[f"{p}_true"]) for r in rows]; m = [f(r[f"{p}_med"]) for r in rows]
    cov = st.mean(f(r[f"{p}_lo"]) <= a <= f(r[f"{p}_hi"]) for r, a in zip(rows, t))
    out = dict(n=len(rows), bias=100 * st.mean((a - b) / b for a, b in zip(m, t)), cov=cov,
               width=st.mean((f(r[f"{p}_hi"]) - f(r[f"{p}_lo"])) / a for r, a in zip(rows, t)))
    s = [f(r.get(f"{p}_{sd}", "nan")) for r in rows]
    if all(not math.isnan(x) for x in s) and st.mean(s) > 0: out["k"] = st.stdev(m) / st.mean(s)
    return out
def pct(x): return f"${x:+.1f}$".replace("+", "{+}")
def cell(mt, k=True):
    if mt is None: return "--"
    s = f"{pct(mt['bias'])}\\,/\\,{mt['cov']:.2f}"
    if k and "k" in mt: s += f"\\,/\\,{mt['k']:.1f}"
    return s
def fmtN(x): return f"{x:,.0f}".replace(",", "{,}")

# ------------------------------------------------------------------ S3
s3 = load(os.path.join(D3, "s3_main.csv"))
if len(s3) >= 4050:
    g = collections.defaultdict(list)
    for r in s3: g[(r["cut"], int(r["n"]), f(r["R0"]), r["model"], f(r["dt"]))].append(r)
    arms = [("exact", 0.0), ("ddsa", 1.0), ("dsa_exact", 0.0), ("dsa_jitter", 1.0)]
    L = ["\\begin{tabular}{l r c c c c c}", "\\toprule",
         "Cut-off & $n$ & $R_0$ & Complete data & DDSA, geometric (daily) & DSA (exact times) & DSA (daily, spread) \\\\", "\\midrule"]
    for cut, lab in (("end", "end"), ("peak", "peak")):
        for n in (100, 1000, 10000):
            for R0 in (1.5, 2.0, 4.0):
                L.append(f"{lab} & {fmtN(n)} & {R0:g} & " + " & ".join(cell(met(g[(cut, n, R0, a, dt)], "R0")) for a, dt in arms) + " \\\\")
        if cut == "end": L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("s3_R0.tex", "\n".join(L) + "\n")
    # Δt convergence (R0 mean relative bias / coverage, n = 1,000, end cut-off); then mean IP by arm
    L = ["\\begin{tabular}{c c c c c c c}", "\\toprule",
         "True $R_0$ & DDSA $\\Delta t=1$ & $\\Delta t=0.5$ & $\\Delta t=0.25$ & $\\Delta t=0.1$ & DSA (exact times) & Complete data \\\\", "\\midrule"]
    for R0 in (1.5, 2.0, 4.0):
        row = [cell(met(g[("end", 1000, R0, "ddsa", dt)], "R0"), k=False) for dt in (1.0, 0.5, 0.25, 0.1)]
        row += [cell(met(g[("end", 1000, R0, "dsa_exact", 0.0)], "R0"), k=False), cell(met(g[("end", 1000, R0, "exact", 0.0)], "R0"), k=False)]
        L.append(f"{R0:g} & " + " & ".join(row) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("s3_dt.tex", "\n".join(L) + "\n")
    L = ["\\begin{tabular}{l r c c c c c}", "\\toprule",
         "Cut-off & $n$ & $R_0$ & Complete data & DDSA, geometric (daily) & DSA (exact times) & DSA (daily, spread) \\\\", "\\midrule"]
    for cut in ("end", "peak"):
        for n in (100, 1000, 10000):
            for R0 in (1.5, 2.0, 4.0):
                L.append(f"{cut} & {fmtN(n)} & {R0:g} & " + " & ".join(cell(met(g[(cut, n, R0, a, dt)], "μ"), k=False) for a, dt in arms) + " \\\\")
        if cut == "end": L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("s3_meanIP.tex", "\n".join(L) + "\n")
else:
    for nm in ("s3_R0.tex", "s3_dt.tex", "s3_meanIP.tex"): pending(nm, "S3")

# ------------------------------------------------------------------ S4
s4 = load(os.path.join(D3, "s4_main.csv"))
if len(s4) >= 540:
    g = collections.defaultdict(list)
    for r in s4: g[(r["cut"], int(r["n"]), f(r["R0"]), r["model"])].append(r)
    L = ["\\begin{tabular}{l r c c c c c c c}", "\\toprule",
         "& & & \\multicolumn{3}{c}{$N$: bias\\,/\\,coverage} & \\multicolumn{3}{c}{$R_0$: bias\\,/\\,coverage} \\\\",
         "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}",
         "Cut-off & $n$ & $R_0$ & DDSA & DSA, $n$ imputed & DSA, $n$ summed & DDSA & DSA, $n$ imputed & DSA, $n$ summed \\\\", "\\midrule"]
    for cut in ("end", "peak"):
        for n in (100, 1000, 10000):
            for R0 in (1.5, 2.0, 4.0):
                rr = [g[(cut, n, R0, a)] for a in ("ddsa_N", "dsa_sds", "dsa_full")]
                L.append(f"{cut} & {fmtN(n)} & {R0:g} & " + " & ".join(cell(met(x, "N"), k=False) for x in rr) + " & "
                         + " & ".join(cell(met(x, "R0"), k=False) for x in rr) + " \\\\")
        if cut == "end": L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("s4_N.tex", "\n".join(L) + "\n")
else:
    pending("s4_N.tex", "S4")

# ------------------------------------------------------------------ S5
s5 = load(os.path.join(D3, "s5_main.csv")); py = os.path.join(D3, "s6_dsa_python.jsonl")
if len(s5) >= 180:
    g = collections.defaultdict(list)
    for r in s5: g[(r["scenario"], r["model"])].append(r)
    L = ["\\begin{tabular}{c l c c c c c}", "\\toprule",
         "Scenario & Model & $R_0$ & Mean IP & IP SD & Shape (median) & $r_{\\max}$ (median) \\\\", "\\midrule"]
    names = {"ddsa": "DDSA, GLCT (NegBin DPH)", "ddsa_box": "DDSA, Boxcar (discretised Gamma)", "dsa": "DSA, Gamma"}
    for sc in ("A", "B", "C"):
        for mdl in ("ddsa", "ddsa_box", "dsa"):
            rs = g[(sc, mdl)]
            if not rs: continue
            shape = st.median(f(r["shape_med"]) for r in rs)
            rmax = f"{st.median(f(r['rmax']) for r in rs):.0f}" if mdl == "ddsa" else "--"
            r0 = "--" if sc == "C" else cell(met(rs, "R0"))
            L.append(f"{sc} & {names[mdl]} & {r0} & {cell(met(rs, 'μ'))} & {cell(met(rs, 'sd'), k=False)} & {shape:.1f} & {rmax} \\\\")
        if sc != "C": L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("s5_nonexp.tex", "\n".join(L) + "\n")
else:
    pending("s5_nonexp.tex", "S5")

# ------------------------------------------------------------------ S6
b = load(os.path.join(D3, "s6_benchmarks.csv"))
if b:
    names = {"ddsa": "DDSA, geometric", "dsa": "DSA", "ddsa_N": "DDSA, geometric, $N$ summed", "dsa_full": "DSA, $n$ summed",
             "dsa_sds": "DSA, $n$ imputed (MH)", "ddsa_dph": "DDSA, GLCT (NegBin DPH)", "ddsa_box": "DDSA, Boxcar (discretised Gamma)", "dsa_gamma": "DSA, Gamma (renewal grid)"}
    L = ["\\begin{tabular}{l l r r r r r r}", "\\toprule",
         "Section & Model & $n$ & $K$ & Log-lik.\\ (ms) & Gradient (ms) & Fit (s) & ESS/s \\\\", "\\midrule"]
    for r in b:
        L.append(f"{r['block']} & {names.get(r['arm'], r['arm'])} & {fmtN(f(r['n']))} & {fmtN(f(r['K']))} & {f(r['loglik_ms']):.3g} & "
                 f"{'--' if math.isnan(f(r['gradient_ms'])) else '%.3g' % f(r['gradient_ms'])} & {f(r['fit_s']):.1f} & {f(r['ess_per_s']):.3g} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("s6_cost.tex", "\n".join(L) + "\n")
else:
    pending("s6_cost.tex", "S6 benchmarks")
sds = load(os.path.join(D3, "s6_sds_R.csv"))
if sds:
    import re
    g = collections.defaultdict(list)
    for r in sds:
        m = re.match(r"sds_n1000_R0([\d.]+)_(\w+)_rep(\d+)\.csv", r["file"]); g[(m.group(2), float(m.group(1)), r["fitn"])].append(r)
    L = ["\\begin{tabular}{l c c c c c}", "\\toprule",
         "Cut-off & $R_0$ & $n$ & $R_0$ estimate, range of medians & $N$, range of medians & Fit (s) \\\\", "\\midrule"]
    for cut in ("end", "peak"):
        for R0 in (1.5, 2.0, 4.0):
            for fitn, lab in (("FALSE", "known"), ("TRUE", "imputed")):
                rs = g[(cut, R0, fitn)]; r0 = [f(r["R0_med"]) for r in rs]
                Nm = "--" if fitn == "FALSE" else f"{fmtN(min(f(r['N_med']) for r in rs))}--{fmtN(max(f(r['N_med']) for r in rs))}"
                cov = sum(f(r["R0_lo"]) <= R0 <= f(r["R0_hi"]) for r in rs)
                L.append(f"{cut} & {R0:g} & {lab} & {min(r0):.2f}--{max(r0):.2f} ({cov}/{len(rs)} cover) & {Nm} & {st.median(f(r['time_s']) for r in rs):.0f} \\\\")
        if cut == "end": L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("s6_sds.tex", "\n".join(L) + "\n")
if os.path.exists(py):
    rows = [json.loads(l) for l in open(py)]
    L = ["\\begin{tabular}{c c c c c c}", "\\toprule", "Scenario & $R_0$ (true 2.25) & Mean IP (true 9) & IP SD (true 2.45) & Shape (true 13.5) & Fit (s) \\\\", "\\midrule"]
    for sc in ("A", "B"):
        rs = [r for r in rows if r["scen"] == sc]
        rg = lambda k, d: f"{min(r[k] for r in rs):.{d}f}--{max(r[k] for r in rs):.{d}f}"
        L.append(f"{sc} & {rg('R0', 2)} & {rg('mu', 1)} & {rg('sd', 2)} & {rg('shape', 1)} & {st.median(r['time_s'] for r in rs):.0f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("s6_dsa_python.tex", "\n".join(L) + "\n")

# ------------------------------------------------------------------ S7
s7 = load(os.path.join(SA, "S7_stress_tests", "out", "stress_main.csv"))
if len(s7) >= 1500:
    import re
    g = collections.defaultdict(list)
    for r in s7: g[(r["factor"], r["level"])].append(r)
    order = re.findall(r'\("([A-Za-z_]+)", "([^"]+)"\)', open(os.path.join(SA, "S7_stress_tests", "stress_run.jl")).read())   # LEVELS order
    def flagged(r):
        return (f(r["p_inc"]) < 0.05 or (not math.isnan(f(r["p_delta"])) and f(r["p_delta"]) < 0.05) or f(r["p_K"]) < 0.05
                or f(r["rhat"]) > 1.01 or f(r["div"]) > 0)
    base = g[("baseline", "baseline")]; bm = met(base, "R0", sd="sd"); bflag = st.mean(flagged(r) for r in base)
    fac = {"baseline": "Baseline", "resolution": "Recording resolution", "misrecorded": "Mis-recorded 1-d intervals",
           "missing_recovery": "Missing recoveries", "window": "Observation window", "ascertainment": "Ascertainment",
           "unknown_N": "Unknown $N$", "beta_t": "Time-varying $\\beta$", "mixing": "Heterogeneous mixing",
           "superspreading": "Superspreading", "importations": "Importations", "near_critical": "Near-critical"}
    L = ["\\begin{tabular}{l l c c c c l}", "\\toprule",
         "Factor & Level & $R_0$: bias\\,/\\,cov.\\,/\\,$k$ & Mean IP: bias\\,/\\,cov. & $\\hat r$ & Flagged & Class \\\\", "\\midrule"]
    last = None
    for key in order:
        rs = g[key]; mr = met(rs, "R0", sd="sd"); mi = met(rs, "ip", sd="sd"); fl = st.mean(flagged(r) for r in rs)
        biased = abs(mr["bias"]) > 5 or mr["cov"] < bm["cov"] - 0.15
        kind = "biased" if abs(mr["bias"]) > 5 else "miscalibrated"     # the rule fails on bias, or on coverage only
        cls = "--" if key[0] == "baseline" else ("unbiased" if not biased else (f"{kind}, detected" if fl >= bflag + 0.25 else f"{kind}, silent"))
        lab = fac.get(key[0], key[0]) if key[0] != last else ""; last = key[0]
        lev = key[1].replace("%", "\\%").replace("R0", "$R_0$").replace("k = ", "dispersion ").replace("; N ", ", $n=$").replace(";", ",")
        lev = re.sub(r"\$n=\$(\d+)", lambda m: "$n=" + fmtN(int(m.group(1))) + "$", lev)
        L.append(f"{lab} & {lev} & {cell(mr)} & {cell(mi, k=False)} & {st.median(f(r['r_med']) for r in rs):.0f} & {fl:.2f} & {cls} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("s7_stress.tex", "\n".join(L) + "\n")
else:
    pending("s7_stress.tex", "S7 stress tests")
# S7 baseline grid (existing chain-binomial coverage study)
cb = load(os.path.join(SA, "S7_stress_tests", "baseline_grid", "out", "results_cb.csv"))
if cb:
    L = ["\\begin{tabular}{r c c c c}", "\\toprule", "$n$ & True $R_0$ & $R_0$: bias\\,/\\,coverage & Mean IP: bias\\,/\\,coverage & Geometric model, $R_0$: bias\\,/\\,coverage \\\\", "\\midrule"]
    for N in (100, 1000, 10000):
        for R0 in (1.3, 2.0, 3.0):
            sel = lambda m: [x for x in cb if int(x["N"]) == N and f(x["R0"]) == R0 and int(x["r"]) == 4 and x["model"] == m]
            d, gg = sel("dph"), sel("geo")
            L.append(f"{fmtN(N)} & {R0:g} & {cell(met(d, 'R0'), k=False)} & {cell(met(d, 'ip'), k=False)} & {cell(met(gg, 'R0'), k=False)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("s7_grid.tex", "\n".join(L) + "\n")
