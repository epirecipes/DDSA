# Writes the LaTeX tables of supplement Sections S2–S4 from the result files, so the manuscript numbers stay in sync
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
def cols(mt, k=True):
    """Bias (%), coverage and, with k=True, the spread ratio k of a set of fits, as separate table cells."""
    if mt is None: return ["--"] * (3 if k else 2)
    c = [pct(mt["bias"]), f"{mt['cov']:.2f}"]
    if k: c.append(f"{mt['k']:.1f}" if "k" in mt else "--")
    return c
def fmtN(x): return f"{x:,.0f}".replace(",", "{,}")
def panel(ncol, text): return f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{text}}}}} \\\\"   # panel or group heading row
def stack(*lines): return "\\begin{tabular}[b]{@{}c@{}}" + "\\\\".join(lines) + "\\end{tabular}"   # two-line column header

# ------------------------------------------------------------------ S2: larger populations (fitted trajectories)
import re
S2O = os.path.join(SA, "S2_identifiability", "out")
def s2lines(fn): return open(os.path.join(S2O, fn)).read().splitlines() if os.path.exists(os.path.join(S2O, fn)) else []
def qq(line, key):
    m = re.search(re.escape(key) + r"\s*=\s*([-0-9.]+) \[([-0-9.]+), ([-0-9.]+)\]", line)
    return None if m is None else tuple(float(x) for x in m.groups())
def fq(t, d=2): return "--" if t is None else f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]"
def fint(t): return "--" if t is None else f"{fmtN(t[0])} [{fmtN(t[1])}, {fmtN(t[2])}]"
traj = {N: s2lines(f"s2_9_trajectories_N{N}.txt") for N in (99, 1000, 10000)}
if all(traj.values()) and all(any(l.strip().startswith("Infection-only DDSA") for l in L) for L in traj.values()):
    L = ["\\begin{tabular}{r l c c c}", "\\toprule", "$N$ & Model & $R_0$ & Mean infectious period (d) & Peak number infectious \\\\", "\\midrule"]
    grids = (("Map--Poisson", "Map–Poisson", "s2_8_grid_map_poisson_N"), ("ODE--Poisson", "ODE–Poisson", "s2_8_grid_ode_poisson_N"),
             ("Infection-only DDSA", "Infection-only DDSA", "s2_8_grid_infonly_N"))
    for N, T in traj.items():
        tr = next(l for l in T if l.strip().startswith("truth:"))
        R0t = float(re.search(r"R0=([0-9.]+)", tr).group(1)); EWt = float(re.search(r"E\[W\]=([0-9.]+)", tr).group(1))
        pkt = float(re.search(r"number infectious peak ([0-9.]+)", tr).group(1))
        L.append(f"{fmtN(N)} & Truth & {R0t:.2f} & {EWt:.2f} & {fmtN(pkt)} \\\\")
        def band(name):
            i = next(i for i, l in enumerate(T) if l.strip().startswith(name))
            m = re.search(r"number infectious peak ([0-9.]+) \[([0-9.]+), ([0-9.]+)\]", T[i + 1]); return tuple(float(x) for x in m.groups())
        for tex, name, stem in grids:
            ex = next(l for l in s2lines(f"{stem}{N}.txt") if l.strip().startswith("posterior:"))
            L.append(f" & {tex} & {fq(qq(ex, 'R0'))} & {fq(qq(ex, 'meanIP'))} & {fint(band(name))} \\\\")
        fd = next(l for l in T if l.strip().startswith("Full-data DDSA:"))
        L.append(f" & Full-data DDSA & {fq(qq(fd, 'R0'))} & {fq(qq(fd, 'meanIP'))} & {fint(band('Full-data DDSA:'))} \\\\")
        if N != 10000: L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("s2_traj.tex", "\n".join(L) + "\n")

# ------------------------------------------------------------------ S3
s3 = load(os.path.join(D3, "s3_main.csv"))
if len(s3) >= 4050:
    g = collections.defaultdict(list)
    for r in s3: g[(r["cut"], int(r["n"]), f(r["R0"]), r["model"], f(r["dt"]))].append(r)
    arms = [("exact", 0.0), ("ddsa", 1.0), ("dsa_exact", 0.0), ("dsa_jitter", 1.0)]
    head = [stack("Complete", "data"), stack("DDSA", "(daily)"), stack("DSA", "(exact)"), stack("DSA", "(daily)")]
    def s3_table(p, panels, arms=arms, head=head):
        """One panel per summary (index into cols(): 0 bias, 1 coverage, 2 k), with the two cut-offs side by side."""
        a = len(arms); nc = 2 + 2 * a
        L = [f"\\begin{{tabular}}{{r c {'c' * a} {'c' * a}}}", "\\toprule",
             f"& & \\multicolumn{{{a}}}{{c}}{{End cut-off}} & \\multicolumn{{{a}}}{{c}}{{Peak cut-off}} \\\\",
             f"\\cmidrule(lr){{3-{2 + a}}}\\cmidrule(lr){{{3 + a}-{nc}}}",
             "$n$ & True $R_0$ & " + " & ".join(head * 2) + " \\\\"]
        for i, title in panels:
            L += ["\\midrule", panel(nc, title)]
            for n in (100, 1000, 10000):
                for R0 in (1.5, 2.0, 4.0):
                    L.append(f"{fmtN(n)} & {R0:g} & " + " & ".join(cols(met(g[(cut, n, R0, a, dt)], p))[i]
                                                                  for cut in ("end", "peak") for a, dt in arms) + " \\\\")
                if n != 10000: L.append("\\addlinespace")
        L += ["\\bottomrule", "\\end{tabular}"]
        return "\n".join(L) + "\n"
    BIAS, COV = (0, "(a) Relative bias of the posterior median (\\%)"), (1, "(b) Coverage of the 95\\% CrI")
    write("s3_R0.tex", s3_table("R0", [BIAS, COV, (2, "(c) Spread ratio $k$")]))
    write("s3_meanIP.tex", s3_table("μ", [BIAS, COV]))        # mean infectious period, same layout
    def paired_table():
        """Supplement Table S10: bias of R0 for complete data, DDSA and DSA (exact), and the paired difference
        DDSA − DSA (exact) on the same outbreaks, as % of the true R0 (mean, 95% normal interval over replicates)
        and in posterior SDs of DSA (exact) (median over replicates). One panel per cut-off."""
        d1 = lambda x: f"{x:.1f}".replace("-0.0", "0.0")   # one decimal, no negative zero
        byrep = {(r["cut"], int(r["n"]), f(r["R0"]), r["model"], f(r["dt"]), r["rep"]): r for r in s3}
        L = ["\\begin{tabular}{r c ccc cc}", "\\toprule",
             "& & \\multicolumn{3}{c}{Bias of the posterior median (\\%)} & \\multicolumn{2}{c}{DDSA $-$ DSA (exact)} \\\\",
             "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}",
             "$n$ & True $R_0$ & " + " & ".join([stack("Complete", "data"), "DDSA", stack("DSA", "(exact)"),
                                               stack("\\% of $R_0$", "{[}95\\% CI{]}"), stack("Posterior", "SDs")]) + " \\\\"]
        for cut, title in (("end", "(a) Cut-off near the end"), ("peak", "(b) Cut-off near the peak")):
            L += ["\\midrule", panel(7, title)]
            for n in (100, 1000, 10000):
                for R0 in (1.5, 2.0, 4.0):
                    d, z = [], []
                    for (c, nn, rr, m, dt, rep), r in byrep.items():
                        if (c, nn, rr, m, dt) != (cut, n, R0, "ddsa", 1.0): continue
                        q = byrep.get((cut, n, R0, "dsa_exact", 0.0, rep))
                        if q is None: continue
                        d.append(100 * (f(r["R0_med"]) - f(q["R0_med"])) / f(r["R0_true"]))
                        z.append((f(r["R0_med"]) - f(q["R0_med"])) / f(q["R0_sd"]))
                    mu, se = st.mean(d), st.stdev(d) / math.sqrt(len(d))
                    b = [pct(met(g[(cut, n, R0, a, dt)], "R0")["bias"]) for a, dt in arms[:3]]
                    L.append(f"{fmtN(n)} & {R0:g} & " + " & ".join(b) +
                             f" & {pct(mu)} $[{d1(mu - 1.96 * se)}, {d1(mu + 1.96 * se)}]$ & ${d1(st.median(z))}$ \\\\")
                if n != 10000: L.append("\\addlinespace")
        L += ["\\bottomrule", "\\end{tabular}"]
        return "\n".join(L) + "\n"
    write("s3_R0_supp.tex", paired_table())   # supplement Table S10
    def dt_table(p):
        """Convergence in the step Δt (n = 1,000, end cut-off): one panel for the bias, one for the coverage."""
        L = ["\\begin{tabular}{c cccc c c}", "\\toprule",
             "& \\multicolumn{4}{c}{DDSA, records and map at step $\\Delta t$ (d)} & & \\\\", "\\cmidrule(lr){2-5}",
             "True $R_0$ & 1 & 0.5 & 0.25 & 0.1 & DSA (exact) & Complete data \\\\"]
        for i, title in (BIAS, COV):
            L += ["\\midrule", panel(7, title)]
            for R0 in (1.5, 2.0, 4.0):
                row = [cols(met(g[("end", 1000, R0, "ddsa", dt)], p), k=False)[i] for dt in (1.0, 0.5, 0.25, 0.1)]
                row += [cols(met(g[("end", 1000, R0, a, 0.0)], p), k=False)[i] for a in ("dsa_exact", "exact")]
                L.append(f"{R0:g} & " + " & ".join(row) + " \\\\")
        L += ["\\bottomrule", "\\end{tabular}"]
        return "\n".join(L) + "\n"
    write("s3_dt_full.tex", dt_table("R0"))                   # bias and coverage at every step
    write("s3_dt_meanIP.tex", dt_table("μ"))                  # the same for the mean infectious period
    def dt_paired_table():
        """Supplement Table S13 (n = 1,000, end cut-off): paired difference DDSA(Δt) − DSA (exact) on the same outbreaks,
        as % of the truth (mean, 95% normal interval over replicates), and coverage at Δt = 1 and 0.1 and for DSA (exact).
        Panel (a) R0, panel (b) mean infectious period."""
        d1 = lambda x: f"{x:.1f}".replace("-0.0", "0.0")
        byrep = {(r["cut"], int(r["n"]), f(r["R0"]), r["model"], f(r["dt"]), r["rep"]): r for r in s3}
        DT = (1.0, 0.5, 0.25, 0.1)
        L = ["\\begin{tabular}{c cccc ccc}", "\\toprule",
             "& \\multicolumn{4}{c}{DDSA $-$ DSA (exact), \\% of truth {[}95\\% CI{]}} & \\multicolumn{3}{c}{Coverage of the 95\\% CrI} \\\\",
             "\\cmidrule(lr){2-5}\\cmidrule(lr){6-8}",
             "True $R_0$ & " + " & ".join(f"$\\Delta t={dt:g}$" for dt in DT) + " & " +
             " & ".join([stack("DDSA", "$\\Delta t=1$"), stack("DDSA", "$\\Delta t=0.1$"), stack("DSA", "(exact)")]) + " \\\\"]
        for p, title in (("R0", "(a) $R_0$"), ("μ", "(b) Mean infectious period")):
            L += ["\\midrule", panel(8, title)]
            for R0 in (1.5, 2.0, 4.0):
                row = []
                for dt in DT:
                    d = []
                    for (c, nn, rr, m, t, rep), r in byrep.items():
                        if (c, nn, rr, m, t) != ("end", 1000, R0, "ddsa", dt): continue
                        q = byrep.get(("end", 1000, R0, "dsa_exact", 0.0, rep))
                        if q is None: continue
                        d.append(100 * (f(r[p + "_med"]) - f(q[p + "_med"])) / f(r[p + "_true"]))
                    mu, se = st.mean(d), st.stdev(d) / math.sqrt(len(d))
                    row.append(f"{pct(mu)} $[{d1(mu - 1.96 * se)}, {d1(mu + 1.96 * se)}]$")
                row += [cols(met(g[("end", 1000, R0, a, dt)], p), k=False)[1]
                        for a, dt in (("ddsa", 1.0), ("ddsa", 0.1), ("dsa_exact", 0.0))]
                L.append(f"{R0:g} & " + " & ".join(row) + " \\\\")
        L += ["\\bottomrule", "\\end{tabular}"]
        return "\n".join(L) + "\n"
    write("s3_dt.tex", dt_paired_table())                     # supplement Table S13
else:
    for nm in ("s3_R0.tex", "s3_dt.tex", "s3_meanIP.tex", "s3_dt_meanIP.tex"): pending(nm, "S3")

# ------------------------------------------------------------------ S5
s5 = load(os.path.join(D3, "s5_main.csv"))
if len(s5) >= 180:
    g = collections.defaultdict(list)
    for r in s5: g[(r["scenario"], r["model"])].append(r)
    L = ["\\begin{tabular}{c l ccc ccc cc c c}", "\\toprule",
         "& & \\multicolumn{3}{c}{$R_0$} & \\multicolumn{3}{c}{Mean IP} & \\multicolumn{2}{c}{IP SD} & & \\\\",
         "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-10}",
         "Scen. & Model & Bias & Cov. & $k$ & Bias & Cov. & $k$ & Bias & Cov. & " + stack("Gamma", "shape") + " & " + stack("$r$", "($r_{\\max}$)") + " \\\\",
         "\\midrule"]
    names = {"ddsa": "DDSA, GLCT (NegBin DPH)", "ddsa_box": "DDSA, Boxcar (Gamma)", "dsa": "DSA (Gamma)"}
    Ls = list(L)                         # rows of non-converged fits are replaced by a note
    for sc in ("A", "B", "C"):
        for mdl in ("ddsa", "ddsa_box", "dsa"):
            rs = g[(sc, mdl)]
            if not rs: continue
            # shape_med is the Gamma shape for the Boxcar and continuous-time models, and the number of stages r for the GLCT model
            shape = st.median(f(r["shape_med"]) for r in rs)
            r0 = ["--"] * 3 if sc == "C" else cols(met(rs, "R0"))
            cells = r0 + cols(met(rs, "μ")) + cols(met(rs, "sd"), k=False)
            cells += (["--", f"{shape:.0f} ({st.median(f(r['rmax']) for r in rs):.0f})"] if mdl == "ddsa" else [f"{shape:.1f}", "--"])
            if st.median(f(r["rhat"]) for r in rs) > 1.01:            # most fits not converged: marked †, summaries not shown
                Ls.append(f"{sc} & {names[mdl]}$^{{\\dagger}}$ & \\multicolumn{{{len(cells)}}}{{c}}{{\\textit{{not converged (median $\\hat R={st.median(f(r['rhat']) for r in rs):.2f}$); not shown}}}} \\\\")
            else:
                Ls.append(f"{sc} & {names[mdl]} & " + " & ".join(cells) + " \\\\")
        if sc != "C": Ls.append("\\addlinespace")
    Ls += ["\\bottomrule", "\\end{tabular}"]
    write("s5_nonexp_supp.tex", "\n".join(Ls) + "\n")
else:
    pending("s5_nonexp_supp.tex", "S5")

# ------------------------------------------------------------------ S4.1: interval calibration (chain-binomial baseline grid, Table S12)
cb = load(os.path.join(SA, "S7_stress_tests", "baseline_grid", "out", "results_cb.csv"))
wm = load(os.path.join(SA, "S7_stress_tests", "baseline_grid", "out", "final_step3_working_model.csv"))
def kwidth(rows, p):     # spread ratio with the posterior SD taken from the 95% interval width (these files store no SD)
    return st.stdev(f(r[f"{p}_med"]) for r in rows) / st.mean((f(r[f"{p}_hi"]) - f(r[f"{p}_lo"])) / 3.92 for r in rows)
if cb:
    # one panel per quantity: the NegBin DPH model on chain-binomial and on working-model data (bias, coverage, k),
    # and the bias of the misspecified geometric model on the chain-binomial data
    L = ["\\begin{tabular}{r c ccc ccc c}", "\\toprule",
         "& & \\multicolumn{3}{c}{Chain-binomial} & \\multicolumn{3}{c}{Working model} & Chain-binomial \\\\",
         "& & \\multicolumn{3}{c}{NegBin DPH} & \\multicolumn{3}{c}{NegBin DPH} & geometric \\\\",
         "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-9}",
         "$n$ & True $R_0$ & Bias & Cov. & $k$ & Bias & Cov. & $k$ & Bias \\\\"]
    for p, title in (("R0", "(a) $R_0$"), ("ip", "(b) Mean infectious period")):
        L += ["\\midrule", panel(9, title)]
        for N in (100, 1000, 10000):
            for R0 in (1.3, 2.0, 3.0):
                sel = lambda m: [x for x in cb if int(x["N"]) == N and f(x["R0"]) == R0 and int(x["r"]) == 4 and x["model"] == m]
                d, gg = sel("dph"), sel("geo")
                w = [x for x in wm if int(x["N"]) == N and f(x["R0"]) == R0]
                wcells = cols(met(w, p), k=False) + [f"{kwidth(w, p):.1f}"] if len(w) > 1 else ["--"] * 3
                cells = cols(met(d, p), k=False) + [f"{kwidth(d, p):.1f}"] + wcells + [pct(met(gg, p)["bias"])]
                L.append(f"{fmtN(N)} & {R0:g} & " + " & ".join(cells) + " \\\\")
            if N != 10000: L.append("\\addlinespace")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("s7_grid.tex", "\n".join(L) + "\n")

# ------------------------------------------------------------------ S3 validation: published codes in their own settings (s6_4–s6_8)
oR, oO = load(os.path.join(D3, "s6_4_sds_own_R.csv")), load(os.path.join(D3, "s6_5_sds_own_ours.csv"))
at = [json.loads(l) for l in open(os.path.join(D3, "s6_6_dsa_own_at.jsonl"))] if os.path.exists(os.path.join(D3, "s6_6_dsa_own_at.jsonl")) else []
if oR and oO and at:
    R = {(r["set"], r["fitn"] == "TRUE"): r for r in oR}
    O = {(r["set"], r["impl"]): r for r in oO}
    def pair(fitn):
        sets = sorted({k[0] for k in R if k[0].startswith("sellke") and k[1] == fitn})
        return [(R[(x, fitn)], O[(x, "alg41_sdsprior" if fitn else "ours_sdsprior")]) for x in sets]
    rng = lambda v: f"{min(v):.2f}--{max(v):.2f}"
    known, imp = pair(False), pair(True)
    dR0k = max(abs(f(a["R0_med"]) - f(b["R0_med"])) for a, b in known)
    ok = [abs(f(a["R0_med"]) - f(b["R0_med"])) <= 0.02 and abs(f(a["N_med"]) / f(b["N_med"]) - 1) <= 0.02 for a, b in imp]
    wR, wO = R[("wsu", True)], O[("wsu", "alg41_sdsprior")]
    fine = [x for x in at if x["grid"] == 10000]
    dll = max(abs(x["llI_python_at_ours"] - x["llI_ours_at_ours"]) for x in fine)
    est = load(os.path.join(D3, "s6_7_dsa_own_mle.csv"))
    ci = lambda r, p: f"{f(r[p + '_med']):.2f} [{f(r[p + '_lo']):.2f}, {f(r[p + '_hi']):.2f}]"
    L = ["\\begin{tabular}{p{0.26\\textwidth} p{0.2\\textwidth} p{0.2\\textwidth} p{0.22\\textwidth}}", "\\toprule",
         "Setting & Published code & Our implementation & Agreement \\\\", "\\midrule",
         f"SDS, Example 2: {len(known)} outbreaks, $n=1{{,}}000$ known & $R_0$ {rng([f(a['R0_med']) for a, _ in known])} & "
         f"$R_0$ {rng([f(b['R0_med']) for _, b in known])} & $R_0$ medians within {dR0k:.3f} \\\\",
         f"SDS, Example 2: the same outbreaks, $n$ imputed (Algorithm~4.1) & $R_0$ {rng([f(a['R0_med']) for a, _ in imp])} & "
         f"$R_0$ {rng([f(b['R0_med']) for _, b in imp])} & $R_0$ within 0.02 and $N$ within 2\\% in {sum(ok)} of {len(ok)} \\\\",
         f"SDS, example 1: WSU, $n$ imputed & $R_0$ {ci(wR, 'R0')}, $N$ {fmtN(f(wR['N_med']))} & $R_0$ {ci(wO, 'R0')}, $N$ {fmtN(f(wO['N_med']))} & -- \\\\",
         f"DSA, synthetic study: {len(fine)} outbreaks, $N=10{{,}}000$, infection times only & $\\ell_I$ at our optimum & $\\ell_I$ at our optimum & within {dll:.2f} log units \\\\",
         "\\bottomrule", "\\end{tabular}"]
    write("s3_own.tex", "\n".join(L) + "\n")
else:
    pending("s3_own.tex", "S3 validation")
ns = load(os.path.join(D3, "s6_8_wsu_nstep.csv"))
if ns:
    nm = {"alg41": "Imputation step of Algorithm~4.1, $n-K\\sim\\mathrm{NegBin}(K,\\tau_\\infty)$, no $\\binom{n}{K}$",
          "full": "Exact conditional draw under the complete Binomial, $n-K\\sim\\mathrm{NegBin}(K+1,q_T)$",
          "summed": "$N$ summed out of the complete Binomial (NUTS)"}
    L = ["\\begin{tabular}{p{0.33\\textwidth} c c c c}", "\\toprule",
         "Population-size step & $N$ & $R_0$ & Mean IP (d) & $\\hat R$ \\\\", "\\midrule"]
    for r in ns:
        L.append(f"{nm[r['arm']]} & {fmtN(f(r['N_med']))} [{fmtN(f(r['N_lo']))}, {fmtN(f(r['N_hi']))}] & "
                 f"{f(r['R0_med']):.2f} [{f(r['R0_lo']):.2f}, {f(r['R0_hi']):.2f}] & {f(r['mu_med']):.2f} & {f(r['rhat']):.3f} \\\\")
    # DDSA on the same line list (daily records, discrete map): H1N1_WSU/dsa_WSU_Turing.md, posterior medians
    L += ["\\addlinespace", "DDSA (main text, Section~3(c)) & 14{,}144 & 0.97 & 5.12 & -- \\\\", "\\bottomrule", "\\end{tabular}"]
    write("s3_wsu_nstep.tex", "\n".join(L) + "\n")
else:
    pending("s3_wsu_nstep.tex", "S3 WSU")
