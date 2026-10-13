# Discrete dynamic survival analysis of epidemic case line lists
Sandra Montes (`@slmontes`)
2026-10-12

# Introduction

This notebook is a guide to the supplementary analyses of our manuscript
(Sections S2–S9 of the supplementary material). For each section, we
describe what we tested and how, list the scripts that produce the
results and the command to rerun them, and then show the results that
appear in the supplement. The notebook only reads the saved outputs, so
rendering it takes a few minutes. Rerunning all the analyses takes about
ten hours and is done with `run_method_supplement.sh` (see Section
[S9](#s9-reproducibility)).

We attribute each result to the part of DDSA it comes from: the
likelihood (L), the discrete-time forward map and its step $\Delta t$
(M), the GLCT (Generalized Linear Chain Trick) structure with a NegBin
DPH (discrete phase-type) infectious period (G), the Boxcar structure
(B), or the closed, homogeneous, mean-field assumptions that DDSA shares
with continuous-time DSA. Unless stated otherwise, the continuous-time
DSA results come from our own Julia re-implementations. Where we also
show results from the published code of previous DSA analyses by other
authors, namely the SDS R package of KhudaBukhsh et al. (2020) and the
non-Markovian DSA Python code of Di Lauro et al. (2022), these come from
our runs of their code, in the settings we set for this analysis.

Throughout, $N$ is the whole population at risk, including the initial
infectives. When the two are given separately, $n$ is the number of
individuals initially susceptible and $m$ the number of initial
infectives, so that $N = n + m$. $K$ is the number of cases and $\tau$
the fraction of the $N$ expected to be infected during the observation
period (the attack rate). IP is the infectious period, and $r$ the
number of geometric stages of a NegBin DPH infectious period.

## Table of contents

The table below shows where the code and the saved outputs of each
section are.

| Supplement | Folder | Main scripts | Saved outputs |
|----|----|----|----|
| S2 What the data identify; S8 grid posteriors | `S2_identifiability/` | `s2_model.jl` (shared), `s2_0`–`s2_7` | `out/*.txt`, `figs/`, `chains/` |
| S3 DDSA vs continuous-time DSA | `S3_S6_dsa_comparison/` | `s3_core.jl` (shared), `s3_0_checks.jl`, `s3_1_run.jl` | `out/s3_main.csv` |
| S4 Unknown population size | `S3_S6_dsa_comparison/` | `s4_popsize.jl` | `out/s4_main.csv` |
| S5 Non-exponential infectious periods | `S3_S6_dsa_comparison/` | `s5_nonexp.jl`, `s5_0_checks.jl`, `s5_boxcar_dt.jl` | `out/s5_main.csv`, `out/s5_boxcar_dt.csv` |
| S6 Computational cost | `S3_S6_dsa_comparison/` | `s6_0_export.jl`, `s6_1_benchmarks.jl`, `s6_2_sds.R`, `s6_3_dsa_python.py` | `out/s6_*.csv`, `out/s6_dsa_python.jsonl` |
| S7 Stress tests | `S7_stress_tests/` | `stress_core.jl`, `stress_run.jl`, `stress_check.jl`; baseline grid in `baseline_grid/` | `out/stress_main.csv`; `baseline_grid/out/` |
| Tables of S3–S7 | (this folder) | `make_supplement_tables.py` | `supp_tables/*.tex` (shown in this guide) |

The published code of KhudaBukhsh et
al. ([SDS.Epidemic](https://github.com/cbskust/SDS.Epidemic), in R) and
Di Lauro et
al. ([DSA_refactor](https://github.com/Zkeggia/DSA_refactor), in Python)
that we used for comparison is available from their repositories. We
used commits `3ecb29d` and `26671d6`, unmodified.

The helper functions below read the saved outputs and print them as
tables.

``` julia
using CSV, DataFrames, Printf, Statistics
const SA  = @__DIR__
const S2O = joinpath(SA, "S2_identifiability", "out")
const D3  = joinpath(SA, "S3_S6_dsa_comparison", "out")
const D7  = joinpath(SA, "S7_stress_tests")

"First `(median [lo, hi])` after `key=` or `key = ` in a line of an S2 output file."
function grab(line, key)
    m = match(Regex(key * raw"\s*=\s*([-0-9.]+) \[([-0-9.]+), ([-0-9.]+)\]"), line)
    m === nothing ? "—" : "$(m[1]) [$(m[2]), $(m[3])]"
end
"Lines of an S2 output file whose start matches `prefix`."
lines_starting(file, prefix) = filter(l -> startswith(strip(l), prefix), readlines(joinpath(S2O, file)))
"Names of the six S2 models. The S2 scripts and output files number them 1–6 (`rung 1` … `rung 6`)."
const MODEL = Dict("1" => "infection-only DDSA", "2" => "map–Poisson", "3" => "map–NegBin",
                   "4" => "ODE–Poisson", "5" => "ODE–NegBin", "6" => "full-data DDSA")
"Replace `rung k` / `Rung k` in a line of an S2 output file with the model name."
withnames(l) = replace(replace(l, r"[Rr]ung (\d)" => s -> (n = MODEL[string(s[end])]; s[1] == 'R' ? uppercasefirst(n) : n)),
                       r"\((=|≡)(?! )" => s"(\1 ")

const TABLES = joinpath(SA, "supp_tables")
"Print a supplement table (written by make_supplement_tables.py) as a markdown table with the given column headers."
function tex_table(file, header)
    L = readlines(joinpath(TABLES, file))
    body = L[findfirst(l -> occursin("\\midrule", l), L)+1:end]
    rows = [strip(replace(l, r"\\\\\s*$" => "")) for l in body if endswith(strip(l), "\\\\")]
    clean(c) = strip(replace(c, "\\,/\\," => " / ", "{+}" => "+", "{,}" => ",", "--" => "–", "\\%" => "%"))
    mdrow(header); mdrow(fill("---", length(header)))
    for r in rows
        mdrow(clean.(split(r, "&")))
    end
    println("\n*Supplement table file: `supp_tables/$file`.*")
end

"Convergence of a set of fits: number with R̂ > 1.01 and with at least one divergent transition."
conv(df) = println("*Convergence: $(nrow(df)) fits; $(count(>(1.01), df.rhat)) with R̂ > 1.01, $(count(>(0), df.div)) with at least one divergent transition (all kept in the summaries).*")

"Relative bias (%) and 95% CrI coverage of the posterior median over replicates, as in the supplement tables."
function bias_cov(df, p)
    t = df[!, "$(p)_true"]; m = df[!, "$(p)_med"]
    (bias = 100 * mean((m .- t) ./ t), cov = mean(df[!, "$(p)_lo"] .<= t .<= df[!, "$(p)_hi"]))
end
mdrow(cells) = println("| ", join(cells, " | "), " |")
```

# S1 Prior specifications

Supplement S1 (Tables S1–S5) lists the priors of every main-text
analysis. These are the priors defined in the `@model` blocks of the
case-study notebooks, where the integer-valued parameters ($N$, $r$) are
summed out of the likelihood.

# S2 What the data identify

In Section 3(b) of the manuscript, the count-based models are fitted
only to the daily incidence counts (the number of new infections per
day, with a Poisson or Negative-Binomial observation model), whereas
DDSA is fitted to the full line list: each case’s infection day and
recovery interval, and the final size out of the $N = 99$ at risk. With
the removal rate estimated, the count-based models leave $R_0$ almost as
uncertain as its prior, whereas DDSA does not. The two approaches differ
in several ways at once, so in this section we change one factor at a
time, building an identifiability ladder. All models use the synthetic
outbreak of Sections 3(a)–(b) (Erlang-4 DPH, $\beta = 0.5$, $N = 99$,
$T = 40$ d, $K = 91$ cases; true $R_0 = 3.164$ and mean infectious
period 6.33 d) and the same priors: $R_0 \sim \mathrm{U}(0.5, 5)$, mean
IP $\mu \sim \mathrm{U}(1, 30)$ and $\rho \sim \mathrm{Beta}(1, N-1)$,
with $r$ summed out over $1, \ldots, 10$.

The ladder has six models, each fitted to this outbreak:

1.  **Infection-only DDSA** (Scenario B of manuscript Section 3(a)). The
    discrete map, with the infection days scored by the Multinomial
    timing term and the number of cases by the Binomial final-size term.
2.  **Map–Poisson.** The discrete map, with the daily incidence counts
    scored by independent Poisson terms.
3.  **Map–NegBin.** As map–Poisson, with Negative Binomial terms.
4.  **ODE–Poisson.** As map–Poisson, with an Erlang-$r$ ODE in place of
    the discrete map.
5.  **ODE–NegBin.** As map–NegBin, with the Erlang-$r$ ODE.
6.  **Full-data DDSA** (Scenario A of manuscript Section 3(a)).
    Infection-only DDSA plus the recovery-interval term $\ell_R$.

The first five models use no recovery intervals. The four count models
replace the likelihood of infection-only DDSA with a count likelihood.
The S2 scripts and output files number the models 1–6 in this order
(`rung1` to `rung6`). The tables and outputs below show their names.

The scripts are in `S2_identifiability/`. `s2_model.jl` defines the
data, the forward maps and the likelihood of each model, and
`s2_0_check_likelihoods.jl` checks the infection-only DDSA likelihood
against Scenario B of the data-availability notebook. `s2_1` fits the
six models, the $2\times2$ design and the unknown-$N$ models; `s2_2`
runs the wide-prior checks, the never-infected-cell models and the
Section 3(b) priors; `s2_3` produces the growth-rate ridge (Figure S1)
and the information on $r$; `s2_4` and `s2_5` run the attack-rate sweep
by MCMC and on exact grids (Table S8); and `s2_6` runs the unknown-$N$
grid with a seed prior on the count scale.

``` julia
# from this folder
./run_method_supplement.sh S2
```

## Table S6: the ladder

Each row of the table is one model of the ladder.

``` julia
rungs = Dict(split(l)[1] => l for l in readlines(joinpath(S2O, "s2_1_rungs.txt")) if !isempty(l) && isdigit(l[1]))
wide  = Dict(split(l)[1] => l for f in ("s2_2_corroborations_wide_rungs123.txt", "s2_2_corroborations_wide_rungs456.txt")
             for l in readlines(joinpath(S2O, f)) if !isempty(l) && isdigit(l[1]))
rungname = Dict("1" => "discrete map, Multinomial + Binomial (Scenario B)", "2" => "discrete map, Poisson counts",
             "3" => "discrete map, NegBin counts", "4" => "Erlang-r ODE, Poisson counts",
             "5" => "Erlang-r ODE, NegBin counts", "6" => "discrete map, full line list (Scenario A)")
mdrow(["Model", "Forward model and likelihood", "R0, U(0.5,5)", "Mean IP (d)", "R0, U(0.5,20)"]); mdrow(fill("---", 5))
for r in string.(1:6)
    mdrow([uppercasefirst(MODEL[r]), rungname[r], grab(rungs[r], "R0"), grab(rungs[r], "meanIP"), grab(wide[r], "R0")])
end
println("\n*Source: `S2_identifiability/out/s2_1_rungs.txt`, `s2_2_corroborations_wide_rungs*.txt`. Truth: R0 = 3.16, mean IP 6.33 d.*")
```

| Model | Forward model and likelihood | R0, U(0.5,5) | Mean IP (d) | R0, U(0.5,20) |
|----|----|----|----|----|
| Infection-only DDSA | discrete map, Multinomial + Binomial (Scenario B) | 3.11 \[2.44, 4.44\] | 6.34 \[3.76, 11.2\] | 3.2 \[2.46, 10.33\] |
| Map–Poisson | discrete map, Poisson counts | 4.15 \[2.42, 4.97\] | 8.84 \[4.05, 12.23\] | 10.09 \[3.65, 14.59\] |
| Map–NegBin | discrete map, NegBin counts | 4.19 \[2.43, 4.97\] | 8.77 \[4.08, 12.24\] | 10.22 \[3.93, 15.11\] |
| ODE–Poisson | Erlang-r ODE, Poisson counts | 4.24 \[2.56, 4.97\] | 10.1 \[5.02, 13.53\] | 9.16 \[3.11, 12.9\] |
| ODE–NegBin | Erlang-r ODE, NegBin counts | 4.25 \[2.54, 4.97\] | 9.93 \[4.83, 13.55\] | 9.64 \[3.96, 13.46\] |
| Full-data DDSA | discrete map, full line list (Scenario A) | 3.13 \[2.74, 3.57\] | 6.4 \[6.03, 6.83\] | 3.13 \[2.75, 3.58\] |

*Source: `S2_identifiability/out/s2_1_rungs.txt`,
`s2_2_corroborations_wide_rungs*.txt`. Truth: R0 = 3.16, mean IP 6.33
d.*

With $N$ known and no recovery intervals, $R_0$ is only bounded by its
prior once we replace the Binomial final-size term by count likelihoods
(the four count models). Overdispersion and the continuous-time ODE do
not change this materially. Adding the recovery intervals (full-data
DDSA) narrows the interval further.

## Table S7: isolating the final-size term

Next, we keep the data of infection-only DDSA (infection times and the
final size, no recovery intervals) and change only the likelihood, to
isolate what the final-size term contributes. “Mult” is the Multinomial
timing term of DDSA. The first four rows form the $2\times2$ design:
Multinomial timing or Poisson daily counts, each with or without the
Binomial final-size term. The never-infected-cell rows add one count
cell for the individuals never infected. The last three rows sum $N$
out, with $\rho \sim \mathrm{Beta}(1, 98)$ on the fraction of initial
infectives (the fraction scale) or with an Exp(1) prior on their number
$I_0 = \rho N$ (the count scale).

``` julia
mdrow(["Likelihood (data of infection-only DDSA)", "R0", "Source"]); mdrow(fill("---", 3))
for l in lines_starting("s2_1_2x2.txt", "2x2")
    mdrow([withnames(strip(split(split(l, "|")[1], "2x2")[2])), grab(l, "R0"), "`s2_1_2x2.txt`"])
end
for l in lines_starting("s2_2_corroborations_misc.txt", "R0≤5")
    mdrow(["Never-infected cell, " * replace(strip(split(split(l, "|")[1], "R0≤5")[2]), " + cell" => "", "continuous ODE" => "ODE"), grab(l, "R0"), "`s2_2_corroborations_misc.txt`"])
end
for l in lines_starting("s2_1_unknownN.txt", "1N")
    mdrow([withnames(strip(split(split(l, "|")[1], "1N")[2])), grab(l, "R0") * " (N " * something(match(r"N=([0-9.]+ \[[0-9., ]+\])", l), ["—"])[1] * ")", "`s2_1_unknownN.txt`"])
end
ex = only(lines_starting("s2_6_unknownN_seedprior_grid.txt", "Exact posterior"))
mdrow(["Infection-only DDSA, N ~ U(91, 20,090), seed count I0 = ρN ~ Exp(1) (exact grid)", grab(ex, "R0"), "`s2_6_unknownN_seedprior_grid.txt`"])
```

| Likelihood (data of infection-only DDSA) | R0 | Source |
|----|----|----|
| Mult + Binomial (= infection-only DDSA) | 3.11 \[2.44, 4.44\] | `s2_1_2x2.txt` |
| Mult + Poisson(K;Nτ) (≡ map–Poisson) | 4.19 \[2.52, 4.97\] | `s2_1_2x2.txt` |
| Poisson counts + Binomial | 3.11 \[2.46, 4.52\] | `s2_1_2x2.txt` |
| Poisson counts (= map–Poisson) | 4.15 \[2.42, 4.97\] | `s2_1_2x2.txt` |
| Never-infected cell, discrete map | 3.12 \[2.43, 4.44\] | `s2_2_corroborations_misc.txt` |
| Never-infected cell, ODE | 3.14 \[2.46, 4.46\] | `s2_2_corroborations_misc.txt` |
| Infection-only DDSA, N ~ DiscreteUniform(91, 500) | 4.2 \[2.31, 4.96\] (N 94.0 \[91.0, 107.0\]) | `s2_1_unknownN.txt` |
| Infection-only DDSA, N ~ DiscreteUniform(91, 20090) | 1.1 \[0.5, 4.93\] (N 607.0 \[91.0, 18878.07\]) | `s2_1_unknownN.txt` |
| Infection-only DDSA, N ~ U(91, 20,090), seed count I0 = ρN ~ Exp(1) (exact grid) | 4.175 \[2.225, 4.975\] | `s2_6_unknownN_seedprior_grid.txt` |

When $N$ has a wide support and the seed prior is on the fraction scale,
the chains of this fit end up in two modes. The per-chain output below
shows the second, local mode (a sub-critical outbreak seeded by about 91
infectives) and its lower log-density. IP is the mean infectious period
and $\rho N$ the number of initial infectives:

``` julia
foreach(println, filter(l -> occursin("chain", l), readlines(joinpath(S2O, "s2_1_unknownN.txt")))[6:10])
```

        chain 1: R0=4.24 [2.34, 4.96]  IP=9.11 [3.8, 12.44]  τ=0.97 [0.86, 0.99]  N=94.0 [91.0, 107.0]  ρ=0.01 [0.0, 0.02]  ρN=0.98 [0.37, 2.25]
        chain 2: R0=0.52 [0.5, 0.58]  IP=13.9 [11.47, 16.28]  τ=0.01 [0.0, 0.04]  N=9753.0 [2525.98, 19491.18]  ρ=0.01 [0.0, 0.04]  ρN=91.41 [67.32, 116.21]
        chain 3: R0=0.52 [0.5, 0.59]  IP=13.89 [11.59, 16.16]  τ=0.01 [0.0, 0.04]  N=9763.5 [2421.75, 19437.03]  ρ=0.01 [0.0, 0.04]  ρN=91.53 [68.49, 115.93]
        chain 4: R0=4.12 [2.09, 4.96]  IP=8.69 [3.13, 12.11]  τ=0.97 [0.82, 0.99]  N=94.0 [91.0, 114.0]  ρ=0.01 [0.0, 0.02]  ρN=0.91 [0.34, 2.04]
        chain mean log-joint: [-50.7, -79.3, -79.3, -50.7] | Rhat≤1.735 ESS≥48 div=0 | 281.0 s

## Figure S1: the growth-rate ridge

![Joint posterior draws of $R_0$ and the mean infectious period for
infection-only DDSA, map–Poisson and full-data DDSA, with the discrete
Euler–Lotka relation at the true growth
rate.](S2_identifiability/figs/s2_fig_ridge_R0_meanIP.png)

In the output below, R0≤5 and R0≤20 mark the prior on $R_0$, U(0.5, 5)
or U(0.5, 20), and E\[W\] is the true mean infectious period.

``` julia
foreach(println ∘ withnames, readlines(joinpath(S2O, "s2_3_ridge_r.txt"))[1:8])
```

    True early growth rate λ = 0.3643 /day (doubling time 1.9 d); truth R0 = 3.164, E[W] = 6.328
    Euler–Lotka check: R0_theory(r=4, μ=E[W], λ_true) = 3.164 (NegBin DPH with the truth mean; truth DPH is Erlang-from-mean, so close but not identical)
    R0≤5 map–Poisson: implied growth rate λ = 0.36 [0.3, 0.43] /day; R0 = 4.15 [2.42, 4.97]; mean IP = 8.84 [4.05, 12.23]; corr(log R0, log IP) = 0.896
    R0≤5 infection-only DDSA: implied growth rate λ = 0.36 [0.3, 0.43] /day; R0 = 3.11 [2.44, 4.44]; mean IP = 6.34 [3.76, 11.2]; corr(log R0, log IP) = 0.873
    R0≤5 full-data DDSA: implied growth rate λ = 0.36 [0.31, 0.41] /day; R0 = 3.13 [2.74, 3.57]; mean IP = 6.4 [6.03, 6.83]; corr(log R0, log IP) = 0.253
    R0≤20 map–Poisson: implied growth rate λ = 0.37 [0.32, 0.43] /day; R0 = 10.09 [3.65, 14.59]; mean IP = 22.52 [7.46, 29.66]; corr(log R0, log IP) = 0.959
    R0≤20 infection-only DDSA: implied growth rate λ = 0.35 [0.28, 0.42] /day; R0 = 3.2 [2.46, 10.33]; mean IP = 6.7 [3.77, 27.74]; corr(log R0, log IP) = 0.956
    R0≤20 full-data DDSA: implied growth rate λ = 0.36 [0.31, 0.41] /day; R0 = 3.13 [2.75, 3.58]; mean IP = 6.4 [6.04, 6.83]; corr(log R0, log IP) = 0.271

The infection times fix the growth rate ($\lambda \approx 0.36$ d$^{-1}$
in all three fits), and $R_0$ and the mean infectious period trade off
along it. The recovery intervals then fix where on this ridge the
posterior lies. Without recovery intervals, the shape of the infectious
period is, in most cases, not identified, as the KL divergences from the
prior to the posterior of $r$ show. The effective prior of $r$ is its
uniform prior on 1–10, restricted to $r < \mu$ by the prior on the mean
infectious period:

``` julia
foreach(l -> println(withnames(replace(l, r"(rung \d) [^:]+:" => s"\1:"))),
        filter(l -> occursin("KL", l), readlines(joinpath(S2O, "s2_3_ridge_r.txt"))))
```

    Effective prior P(r) = [0.118, 0.114, 0.11, 0.106, 0.102, 0.098, 0.094, 0.09, 0.086, 0.082]; max possible KL (posterior = point mass at r=4) = 2.243 nats
    R0≤20 infection-only DDSA: P(r) = [0.03, 0.084, 0.142, 0.185, 0.191, 0.142, 0.095, 0.062, 0.04, 0.029] | KL(post‖prior) = 0.162 nats | entropy 2.124 (prior 2.296)
    R0≤20 map–Poisson: P(r) = [0.058, 0.079, 0.094, 0.103, 0.108, 0.111, 0.113, 0.112, 0.112, 0.11] | KL(post‖prior) = 0.04 nats | entropy 2.286 (prior 2.296)
    R0≤20 map–NegBin: P(r) = [0.064, 0.082, 0.094, 0.102, 0.107, 0.11, 0.111, 0.111, 0.11, 0.109] | KL(post‖prior) = 0.034 nats | entropy 2.29 (prior 2.296)
    R0≤20 ODE–Poisson: P(r) = [0.043, 0.068, 0.088, 0.103, 0.113, 0.117, 0.118, 0.117, 0.117, 0.117] | KL(post‖prior) = 0.067 nats | entropy 2.267 (prior 2.296)
    R0≤20 ODE–NegBin: P(r) = [0.05, 0.072, 0.089, 0.1, 0.107, 0.112, 0.115, 0.117, 0.118, 0.118] | KL(post‖prior) = 0.056 nats | entropy 2.276 (prior 2.296)
    R0≤20 full-data DDSA: P(r) = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] | KL(post‖prior) = 2.243 nats | entropy 0.0 (prior 2.296)
    R0≤5  infection-only DDSA: P(r) = [0.031, 0.089, 0.155, 0.203, 0.205, 0.143, 0.09, 0.049, 0.024, 0.012] | KL(post‖prior) = 0.23 nats
    R0≤5  map–Poisson: P(r) = [0.028, 0.062, 0.106, 0.133, 0.147, 0.143, 0.134, 0.114, 0.083, 0.05] | KL(post‖prior) = 0.103 nats
    R0≤5  full-data DDSA: P(r) = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] | KL(post‖prior) = 2.243 nats

## Table S8: the attack-rate sweep (exact grid posteriors)

Finally, we vary the attack rate and compare infection-only DDSA against
map–Poisson, using exact grid posteriors.

``` julia
function sweep_rows(file)
    L = readlines(joinpath(S2O, file)); out = []
    for (i, l) in enumerate(L)
        m = match(r"N=(\d+) R0_true=([0-9.]+) rung (\d)", l)
        m === nothing && continue
        ex = L[i+1]
        push!(out, (N = m[1], R0 = m[2], rung = m[3], est = grab(ex, "R0"),
                    sdtau = parse(Float64, match(r"SD\(τ\) = ([0-9.]+)", ex)[1])))
    end
    out
end
rows = vcat(sweep_rows("s2_5_tausweep_grid_N99.txt"),
            [sweep_rows("s2_5_tausweep_grid_N1000_R0$c.txt") for c in ("1.5", "2.0", "3.16")]...)
mdrow(["N", "True R0", "Infection-only DDSA (Binomial final size)", "Map–Poisson (Poisson counts)", "SD(τ) ratio, map–Poisson / infection-only DDSA"]); mdrow(fill("---", 5))
for N in ("99", "1000"), R0 in ("1.5", "2.0", "3.16")
    r1 = only(filter(x -> x.N == N && x.R0 == R0 && x.rung == "1", rows)); r2 = only(filter(x -> x.N == N && x.R0 == R0 && x.rung == "2", rows))
    mdrow([N, R0, r1.est, r2.est, @sprintf("%.2f", r2.sdtau / r1.sdtau)])
end
println("\n*Source: `S2_identifiability/out/s2_5_tausweep_grid_*.txt`.*")
```

| N | True R0 | Infection-only DDSA (Binomial final size) | Map–Poisson (Poisson counts) | SD(τ) ratio, map–Poisson / infection-only DDSA |
|----|----|----|----|----|
| 99 | 1.5 | 1.49 \[1.35, 1.67\] | 1.53 \[1.33, 1.95\] | 1.60 |
| 99 | 2.0 | 2.15 \[1.85, 2.57\] | 4.3 \[1.95, 7.5\] | 1.62 |
| 99 | 3.16 | 3.21 \[2.48, 10.26\] | 10.06 \[3.86, 14.66\] | 0.39 |
| 1000 | 1.5 | 1.51 \[1.46, 1.56\] | 1.52 \[1.44, 1.6\] | 1.61 |
| 1000 | 2.0 | 2.01 \[1.92, 2.11\] | 2.06 \[1.86, 2.35\] | 2.30 |
| 1000 | 3.16 | 3.17 \[2.93, 3.4\] | 4.61 \[2.72, 12.51\] | 3.57 |

*Source: `S2_identifiability/out/s2_5_tausweep_grid_*.txt`.*

The information from the never-infected individuals matters most when
nearly everyone is infected and the population is small. At moderate
attack rates, the Poisson count model also identifies $R_0$, although
less precisely.

# S3 DDSA and continuous-time DSA on simulated outbreaks

In this section we ask whether DDSA, fitted to daily records with the
full line list and $n$ known, estimates $R_0$ and the mean infectious
period as well as continuous-time DSA fitted to exact event times. We
simulated exact event-driven SIR outbreaks with
$n \in \{100, 1{,}000, 10{,}000\}$ and $R_0 \in \{1.5, 2, 4\}$, observed
up to near the peak or near the end of the outbreak, with 50 replicates
per cell and an exponential infectious period with mean 5 d. Each
outbreak was analysed with a complete-data benchmark (the conjugate
posterior of $\beta$ and $\gamma$ given the whole simulated epidemic, a
reference for what the data can identify), with DDSA (geometric
infectious period, where GLCT and Boxcar coincide), and with our
implementation of continuous-time DSA, both on exact times and on daily
records spread within the day (each event placed uniformly within its
day). We also fitted DDSA with $\Delta t = 0.5$, 0.25 and 0.1 d.

The scripts are in `S3_S6_dsa_comparison/`: `s3_core.jl` contains the
simulator, observation formats, likelihoods, models and complete-data
benchmark; `s3_0_checks.jl` checks the solver and the likelihoods; and
`s3_1_run.jl` runs the fits.

``` julia
./run_method_supplement.sh S3
```

The column headers give the numbers in each cell, in order: bias /
coverage, plus $k$ in Table S10.

- **Bias:** the difference between the posterior median and the true
  value, relative to the true value, in %, averaged over the replicates.
- **Coverage:** the proportion of replicates whose 95% CrI contains the
  true value.
- **Spread ratio $k$:** the SD of the posterior medians over replicates
  divided by the mean posterior SD. It is close to 1 when the intervals
  match the variability between replicates, and above 1 when they are
  too narrow.

The “DSA” columns are our implementation of continuous-time DSA. The
cut-off is where observation ends (near the peak or near the end of the
outbreak), and $n$ is the number initially susceptible, with $m = 0.05n$
initial infectives.

Table S10 shows the results for $R_0$:

``` julia
s3 = CSV.read(joinpath(D3, "s3_main.csv"), DataFrame)
conv(filter(r -> r.model != "exact", s3))
println()
tex_table("s3_R0.tex", ["Cut-off", "n", "True R0", "Complete data: bias % / cov. / k", "DDSA, geometric (daily): bias % / cov. / k", "DSA (exact times): bias % / cov. / k", "DSA (daily, spread): bias % / cov. / k"])
```

*Convergence: 3150 fits; 7 with R̂ \> 1.01, 106 with at least one
divergent transition (all kept in the summaries).*

| Cut-off | n | True R0 | Complete data: bias % / cov. / k | DDSA, geometric (daily): bias % / cov. / k | DSA (exact times): bias % / cov. / k | DSA (daily, spread): bias % / cov. / k |
|----|----|----|----|----|----|----|
| end | 100 | 1.5 | $-8.8$ / 0.94 / 1.4 | $-10.2$ / 0.38 / 4.5 | $-12.8$ / 0.32 / 4.4 | $-12.8$ / 0.32 / 4.4 |
| end | 100 | 2 | $-11.3$ / 0.80 / 1.7 | $-10.5$ / 0.38 / 4.4 | $-12.7$ / 0.44 / 4.2 | $-12.7$ / 0.46 / 4.2 |
| end | 100 | 4 | $+0.7$ / 1.00 / 0.7 | $+9.3$ / 0.82 / 1.3 | $+0.8$ / 0.88 / 1.2 | $+0.7$ / 0.90 / 1.2 |
| end | 1,000 | 1.5 | $-0.9$ / 0.96 / 1.0 | $-0.9$ / 0.36 / 3.7 | $-1.0$ / 0.42 / 3.7 | $-1.1$ / 0.42 / 3.7 |
| end | 1,000 | 2 | $+0.6$ / 0.96 / 1.0 | $+0.7$ / 0.42 / 2.6 | $+0.0$ / 0.52 / 2.4 | $+0.0$ / 0.50 / 2.4 |
| end | 1,000 | 4 | $-0.9$ / 0.96 / 0.9 | $+6.1$ / 0.54 / 1.5 | $-1.2$ / 0.84 / 1.4 | $-1.6$ / 0.82 / 1.4 |
| end | 10,000 | 1.5 | $-0.3$ / 0.96 / 1.1 | $-0.2$ / 0.42 / 3.7 | $-0.1$ / 0.46 / 3.6 | $-0.1$ / 0.44 / 3.6 |
| end | 10,000 | 2 | $-0.1$ / 0.96 / 0.9 | $+0.7$ / 0.62 / 2.2 | $-0.1$ / 0.64 / 2.2 | $-0.1$ / 0.62 / 2.2 |
| end | 10,000 | 4 | $-0.0$ / 0.98 / 0.8 | $+7.6$ / 0.00 / 1.4 | $-0.0$ / 0.92 / 1.3 | $-0.4$ / 0.92 / 1.3 |
| peak | 100 | 1.5 | $+0.1$ / 0.94 / 1.1 | $+3.0$ / 0.64 / 1.9 | $-0.2$ / 0.64 / 1.9 | $-0.1$ / 0.62 / 1.9 |
| peak | 100 | 2 | $-12.5$ / 0.86 / 1.3 | $-12.3$ / 0.70 / 1.9 | $-14.0$ / 0.70 / 1.8 | $-14.4$ / 0.68 / 1.8 |
| peak | 100 | 4 | $+0.9$ / 0.94 / 0.9 | $+10.1$ / 0.90 / 1.1 | $+4.8$ / 0.92 / 1.1 | $+4.3$ / 0.88 / 1.1 |
| peak | 1,000 | 1.5 | $-1.4$ / 0.88 / 1.1 | $-2.0$ / 0.66 / 1.9 | $-2.1$ / 0.64 / 1.9 | $-2.4$ / 0.68 / 1.8 |
| peak | 1,000 | 2 | $-2.4$ / 0.98 / 1.0 | $-2.6$ / 0.64 / 1.9 | $-2.7$ / 0.66 / 1.9 | $-3.0$ / 0.62 / 1.9 |
| peak | 1,000 | 4 | $-1.2$ / 1.00 / 0.9 | $+0.8$ / 0.88 / 1.2 | $-0.7$ / 0.88 / 1.2 | $-1.4$ / 0.86 / 1.3 |
| peak | 10,000 | 1.5 | $-0.5$ / 0.92 / 1.2 | $-0.8$ / 0.70 / 2.0 | $-0.7$ / 0.66 / 2.0 | $-1.1$ / 0.70 / 2.0 |
| peak | 10,000 | 2 | $-0.1$ / 1.00 / 0.9 | $-0.4$ / 0.68 / 1.7 | $-0.3$ / 0.72 / 1.7 | $-0.6$ / 0.70 / 1.7 |
| peak | 10,000 | 4 | $-0.1$ / 0.92 / 1.0 | $+0.7$ / 0.92 / 1.2 | $-0.1$ / 0.90 / 1.2 | $-0.9$ / 0.86 / 1.2 |

*Supplement table file: `supp_tables/s3_R0.tex`.*

Table S11 shows the effect of the step of the map at $n = 1{,}000$ with
the end cut-off ($R_0$ bias / coverage):

``` julia
tex_table("s3_dt.tex", ["True R0", "DDSA Δt = 1: bias % / cov.", "Δt = 0.5: bias % / cov.", "Δt = 0.25: bias % / cov.", "Δt = 0.1: bias % / cov.", "DSA (exact times): bias % / cov.", "Complete data: bias % / cov."])
```

| True R0 | DDSA Δt = 1: bias % / cov. | Δt = 0.5: bias % / cov. | Δt = 0.25: bias % / cov. | Δt = 0.1: bias % / cov. | DSA (exact times): bias % / cov. | Complete data: bias % / cov. |
|----|----|----|----|----|----|----|
| 1.5 | $-0.9$ / 0.36 | $-0.9$ / 0.36 | $-0.9$ / 0.36 | $-0.9$ / 0.36 | $-1.0$ / 0.42 | $-0.9$ / 0.96 |
| 2 | $+0.7$ / 0.42 | $+0.3$ / 0.46 | $+0.1$ / 0.46 | $+0.1$ / 0.46 | $+0.0$ / 0.52 | $+0.6$ / 0.96 |
| 4 | $+6.1$ / 0.54 | $+2.2$ / 0.80 | $+0.6$ / 0.82 | $-0.4$ / 0.80 | $-1.2$ / 0.84 | $-0.9$ / 0.96 |

*Supplement table file: `supp_tables/s3_dt.tex`.*

Table S12 shows the results for the mean infectious period (bias /
coverage):

``` julia
tex_table("s3_meanIP.tex", ["Cut-off", "n", "True R0", "Complete data: bias % / cov.", "DDSA, geometric (daily): bias % / cov.", "DSA (exact times): bias % / cov.", "DSA (daily, spread): bias % / cov."])
```

| Cut-off | n | True R0 | Complete data: bias % / cov. | DDSA, geometric (daily): bias % / cov. | DSA (exact times): bias % / cov. | DSA (daily, spread): bias % / cov. |
|----|----|----|----|----|----|----|
| end | 100 | 1.5 | $-4.7$ / 0.96 | $-2.5$ / 0.80 | $-6.5$ / 0.78 | $-6.0$ / 0.78 |
| end | 100 | 2 | $-6.6$ / 0.96 | $-2.1$ / 0.90 | $-4.9$ / 0.84 | $-4.4$ / 0.86 |
| end | 100 | 4 | $-0.7$ / 0.98 | $+1.8$ / 0.98 | $+1.2$ / 0.96 | $+1.8$ / 0.94 |
| end | 1,000 | 1.5 | $-0.8$ / 0.94 | $+1.5$ / 0.84 | $-1.3$ / 0.82 | $-0.8$ / 0.82 |
| end | 1,000 | 2 | $+0.4$ / 0.94 | $+3.6$ / 0.74 | $+1.5$ / 0.84 | $+2.0$ / 0.80 |
| end | 1,000 | 4 | $-0.1$ / 0.98 | $+1.5$ / 0.90 | $+0.1$ / 0.96 | $+0.7$ / 0.94 |
| end | 10,000 | 1.5 | $-0.4$ / 0.90 | $+2.1$ / 0.50 | $-0.5$ / 0.68 | $+0.0$ / 0.72 |
| end | 10,000 | 2 | $-0.0$ / 0.98 | $+2.5$ / 0.32 | $+0.1$ / 0.88 | $+0.6$ / 0.82 |
| end | 10,000 | 4 | $+0.0$ / 0.94 | $+1.6$ / 0.66 | $+0.0$ / 0.96 | $+0.8$ / 0.88 |
| peak | 100 | 1.5 | $+2.3$ / 0.96 | $+8.5$ / 0.94 | $+8.8$ / 0.94 | $+9.8$ / 0.92 |
| peak | 100 | 2 | $-5.7$ / 0.98 | $-1.3$ / 1.00 | $+0.4$ / 0.96 | $+0.9$ / 0.98 |
| peak | 100 | 4 | $+1.6$ / 0.98 | $-4.6$ / 0.96 | $+9.1$ / 0.96 | $+11.1$ / 0.98 |
| peak | 1,000 | 1.5 | $-1.8$ / 0.90 | $-1.3$ / 0.90 | $-1.1$ / 0.92 | $-0.2$ / 0.92 |
| peak | 1,000 | 2 | $-1.6$ / 0.96 | $-3.5$ / 0.94 | $-1.2$ / 0.98 | $-0.2$ / 0.96 |
| peak | 1,000 | 4 | $-0.2$ / 0.94 | $-11.5$ / 0.48 | $+0.5$ / 0.94 | $+2.3$ / 0.96 |
| peak | 10,000 | 1.5 | $-0.2$ / 0.94 | $-0.5$ / 0.92 | $-0.1$ / 0.96 | $+0.6$ / 0.96 |
| peak | 10,000 | 2 | $+0.1$ / 1.00 | $-2.5$ / 0.74 | $+0.2$ / 1.00 | $+1.1$ / 0.96 |
| peak | 10,000 | 4 | $-0.2$ / 0.94 | $-11.9$ / 0.00 | $-0.1$ / 0.94 | $+1.9$ / 0.90 |

*Supplement table file: `supp_tables/s3_meanIP.tex`.*

The tables are written by `make_supplement_tables.py` from
`out/s3_main.csv`. DDSA and our implementation of continuous-time DSA
both give $R_0$ intervals that are too narrow, while the complete-data
benchmark is calibrated, so this comes from the likelihood (L). The
$R_0$ bias at $R_0 = 4$ and the +2% bias in the mean infectious period
from same-day removals shrink as the step is refined, so they come from
the map (M).

# S4 Unknown population size

When $N$ is not known, we compare three approaches: DDSA, which sums $N$
out of the Binomial final-size term; our re-implementation of the
imputation step of KhudaBukhsh et al. (Algorithm 4.1); and our
continuous-time DSA with $n$ summed out. We used the same outbreaks as
in S3 (replicates 1–10). We also ran the published SDS R package of
KhudaBukhsh et al., with its code unchanged, on replicates 1–5 of the
$n = 1{,}000$ cells.

The script is `S3_S6_dsa_comparison/s4_popsize.jl`, and the results are
in Tables S13 (our implementations) and S14 (the SDS R package of
KhudaBukhsh et al.) of the supplement.

``` julia
./run_method_supplement.sh S4
```

Table S13 gives the bias and coverage for $N = n + m$ and $R_0$, with 10
replicates per row. “$n$ imputed” is our re-implementation of the
imputation step, and “$n$ summed” our continuous-time DSA with $n$
summed out:

``` julia
s4 = CSV.read(joinpath(D3, "s4_main.csv"), DataFrame)
conv(filter(r -> r.model != "dsa_sds", s4)); println("*(NUTS arms; the imputation arm uses our adaptive Metropolis sampler.)*\n")
tex_table("s4_N.tex", ["Cut-off", "n", "True R0", "N, DDSA: bias % / cov.", "N, DSA (n imputed): bias % / cov.", "N, DSA (n summed): bias % / cov.", "R0, DDSA: bias % / cov.", "R0, DSA (n imputed): bias % / cov.", "R0, DSA (n summed): bias % / cov."])
```

*Convergence: 360 fits; 7 with R̂ \> 1.01, 81 with at least one divergent
transition (all kept in the summaries).* *(NUTS arms; the imputation arm
uses our adaptive Metropolis sampler.)*

| Cut-off | n | True R0 | N, DDSA: bias % / cov. | N, DSA (n imputed): bias % / cov. | N, DSA (n summed): bias % / cov. | R0, DDSA: bias % / cov. | R0, DSA (n imputed): bias % / cov. | R0, DSA (n summed): bias % / cov. |
|----|----|----|----|----|----|----|----|----|
| end | 100 | 1.5 | $+13.3$ / 1.00 | $-16.6$ / 1.00 | $+11.7$ / 1.00 | $-10.7$ / 0.90 | $+23.2$ / 0.90 | $-11.4$ / 1.00 |
| end | 100 | 2 | $+26.4$ / 0.80 | $-30.1$ / 0.60 | $+24.3$ / 0.80 | $-8.0$ / 0.50 | $+17.8$ / 0.70 | $-8.9$ / 0.60 |
| end | 100 | 4 | $+2.3$ / 1.00 | $-1.1$ / 0.90 | $+0.1$ / 0.80 | $+7.6$ / 0.90 | $+6.2$ / 0.90 | $+1.7$ / 0.90 |
| end | 1,000 | 1.5 | $+18.2$ / 0.70 | $-5.4$ / 0.40 | $+17.4$ / 0.70 | $-0.2$ / 0.60 | $+4.9$ / 0.60 | $+0.3$ / 0.60 |
| end | 1,000 | 2 | $+4.9$ / 0.80 | $-0.6$ / 1.00 | $+3.0$ / 0.90 | $-3.5$ / 0.60 | $+0.4$ / 0.90 | $-3.5$ / 0.60 |
| end | 1,000 | 4 | $+0.9$ / 0.80 | $-1.2$ / 0.40 | $-0.1$ / 0.90 | $+3.6$ / 0.90 | $+3.3$ / 0.90 | $-0.4$ / 0.90 |
| end | 10,000 | 1.5 | $+1.4$ / 0.60 | $-4.7$ / 0.30 | $+0.7$ / 0.60 | $-1.5$ / 0.30 | $+2.3$ / 0.50 | $-1.1$ / 0.50 |
| end | 10,000 | 2 | $-0.1$ / 0.70 | $-3.1$ / 0.10 | $-0.5$ / 0.80 | $+0.1$ / 0.70 | $+2.9$ / 0.50 | $-0.2$ / 0.70 |
| end | 10,000 | 4 | $+2.0$ / 0.00 | $-1.0$ / 0.00 | $+0.0$ / 1.00 | $+5.0$ / 0.10 | $+4.6$ / 0.10 | $+1.0$ / 0.80 |
| peak | 100 | 1.5 | $+43.1$ / 1.00 | $-73.9$ / 0.00 | $+48.9$ / 1.00 | $+13.3$ / 1.00 | $+192.0$ / 0.00 | $+6.7$ / 1.00 |
| peak | 100 | 2 | $+29.4$ / 1.00 | $-68.0$ / 0.00 | $+29.4$ / 1.00 | $+0.8$ / 1.00 | $+133.9$ / 0.10 | $+0.4$ / 1.00 |
| peak | 100 | 4 | $+46.7$ / 1.00 | $-47.3$ / 0.00 | $+75.1$ / 1.00 | $+13.7$ / 1.00 | $+110.5$ / 0.10 | $-6.6$ / 1.00 |
| peak | 1,000 | 1.5 | $+27.7$ / 1.00 | $-74.3$ / 0.00 | $+32.1$ / 1.00 | $+9.2$ / 1.00 | $+161.8$ / 0.00 | $+6.5$ / 1.00 |
| peak | 1,000 | 2 | $+39.4$ / 1.00 | $-61.2$ / 0.00 | $+40.1$ / 1.00 | $-7.5$ / 1.00 | $+118.2$ / 0.00 | $-8.5$ / 1.00 |
| peak | 1,000 | 4 | $+12.6$ / 0.90 | $-48.7$ / 0.00 | $+26.3$ / 0.90 | $+10.7$ / 0.90 | $+107.2$ / 0.00 | $-1.1$ / 0.90 |
| peak | 10,000 | 1.5 | $+53.1$ / 1.00 | $-75.3$ / 0.00 | $+52.1$ / 1.00 | $-8.5$ / 1.00 | $+159.0$ / 0.00 | $-8.7$ / 1.00 |
| peak | 10,000 | 2 | $+9.7$ / 0.80 | $-62.1$ / 0.00 | $+14.3$ / 0.90 | $-0.5$ / 0.90 | $+119.3$ / 0.00 | $-2.7$ / 0.90 |
| peak | 10,000 | 4 | $-12.3$ / 0.40 | $-45.4$ / 0.00 | $-0.2$ / 1.00 | $+13.7$ / 0.40 | $+100.7$ / 0.00 | $+1.8$ / 1.00 |

*Supplement table file: `supp_tables/s4_N.tex`.*

Table S14 gives the results of our runs of the SDS R package of
KhudaBukhsh et al., with the code unchanged, on replicates 1–5 of the
$n = 1{,}000$ cells. With $n$ known, the never-infected individuals are
included in the data. “$j$/5 cover” is the number of the five replicates
whose 95% interval contains the true value:

``` julia
tex_table("s6_sds.tex", ["Cut-off", "True R0", "n known or imputed", "R0 estimate, range of medians", "N, range of medians", "Fit (s)"])
```

| Cut-off | True R0 | n known or imputed | R0 estimate, range of medians | N, range of medians | Fit (s) |
|----|----|----|----|----|----|
| end | 1.5 | known | 1.33–1.63 (2/5 cover) | – | 232 |
| end | 1.5 | imputed | 1.57–1.87 (1/5 cover) | 720–1,068 | 204 |
| end | 2 | known | 1.85–2.04 (4/5 cover) | – | 163 |
| end | 2 | imputed | 1.90–2.08 (5/5 cover) | 1,007–1,100 | 156 |
| end | 4 | known | 3.84–4.20 (5/5 cover) | – | 94 |
| end | 4 | imputed | 4.02–4.22 (5/5 cover) | 1,030–1,050 | 87 |
| peak | 1.5 | known | 1.42–1.74 (4/5 cover) | – | 44 |
| peak | 1.5 | imputed | 3.61–4.24 (0/5 cover) | 266–332 | 43 |
| peak | 2 | known | 1.79–2.00 (5/5 cover) | – | 46 |
| peak | 2 | imputed | 4.09–4.37 (0/5 cover) | 382–439 | 42 |
| peak | 4 | known | 3.80–4.41 (5/5 cover) | – | 36 |
| peak | 4 | imputed | 7.97–8.37 (0/5 cover) | 500–615 | 34 |

*Supplement table file: `supp_tables/s6_sds.tex`.*

# S5 Non-exponential infectious periods

Here we use the synthetic setting of Di Lauro et al. (a Gamma infectious
period with mean 9 d and shape 13.5, and $N = 10{,}000$) to see what the
GLCT (NegBin DPH) and Boxcar structures recover from daily data, with
the full line list (A), infection times only (B) or recovery intervals
only (C). We compare them with our implementation of continuous-time DSA
with a Gamma infectious period, and with the published non-Markovian DSA
Python code of Di Lauro et al. (maximum likelihood) on replicates 1–5.

The scripts are in `S3_S6_dsa_comparison/`: `s5_nonexp.jl` runs the main
fits; `s5_0_checks.jl` checks the convergence of the renewal grid,
compares the discretised Gamma with the simulated intervals and looks at
the shortest intervals (log in `out/logs/s5_0_checks.log`); and
`s5_boxcar_dt.jl` refits the Boxcar arm at $\Delta t = 0.5$ and 0.25 d,
and is run separately. The results are in Tables S15 and S16 of the
supplement.

``` julia
./run_method_supplement.sh S5
DTS=0.5,0.25 REPS=20 SCEN=A BUDGET=30000 RUN=s5_boxcar_dt julia --project=@. -t 10 S3_S6_dsa_comparison/s5_boxcar_dt.jl
```

Table S15 summarises 20 replicates per row. The true values are
$R_0 = 2.25$, a mean infectious period of 9 d and an SD of 2.45 d. Each
$R_0$ and mean-IP cell gives three numbers, bias / coverage / $k$, and
each IP SD cell gives the first two:

- **Bias:** the difference between the posterior median and the true
  value, relative to the true value, in %, averaged over the replicates.
- **Coverage:** the proportion of replicates whose 95% CrI contains the
  true value.
- **Spread ratio $k$:** the SD of the posterior medians over replicates
  divided by the mean posterior SD. It is close to 1 when the intervals
  match the variability between replicates, and above 1 when they are
  too narrow.

For example, the first row’s $R_0$ cell, +10.4 / 0.00 / 5.5, means that
the posterior median of $R_0$ was on average 10.4% above 2.25, that none
of the 20 intervals contained 2.25, and that the medians varied 5.5
times more between replicates than the posterior SDs indicate. The shape
is the posterior median of the Gamma shape ($r$ for the NegBin DPH), and
$r_{\max}$ is the largest $r$ allowed, the shortest recorded interval.
Both are medians over replicates. The NegBin DPH fits of scenario B did
not converge:

``` julia
conv(CSV.read(joinpath(D3, "s5_main.csv"), DataFrame))
println()
tex_table("s5_nonexp.tex", ["Scenario", "Model", "R0: bias % / cov. / k", "Mean IP: bias % / cov. / k", "IP SD: bias % / cov.", "Shape (median)", "r_max = shortest interval (median)"])
```

*Convergence: 180 fits; 30 with R̂ \> 1.01, 22 with at least one
divergent transition (all kept in the summaries).*

| Scenario | Model | R0: bias % / cov. / k | Mean IP: bias % / cov. / k | IP SD: bias % / cov. | Shape (median) | r_max = shortest interval (median) |
|----|----|----|----|----|----|----|
| A | DDSA, GLCT (NegBin DPH) | $+10.4$ / 0.00 / 5.5 | $-3.0$ / 0.05 / 3.9 | $+104.3$ / 0.00 | 2.0 | 2 |
| A | DDSA, Boxcar (discretised Gamma) | $+3.8$ / 0.00 / 2.7 | $-0.3$ / 0.85 / 1.1 | $-0.2$ / 0.90 | 13.5 | – |
| A | DSA, Gamma | $-0.3$ / 0.40 / 2.7 | $+0.0$ / 0.95 / 1.1 | $+0.3$ / 0.90 | 13.5 | – |
| B | DDSA, GLCT (NegBin DPH) | $+0.4$ / 0.80 / 1.2 | $-12.5$ / 0.10 / 0.8 | $+9.2$ / 0.85 | 4.0 | 10 |
| B | DDSA, Boxcar (discretised Gamma) | $+0.2$ / 0.85 / 1.3 | $-16.5$ / 0.05 / 1.8 | $+19.8$ / 0.70 | 6.5 | – |
| B | DSA, Gamma | $+0.3$ / 0.85 / 1.3 | $-0.1$ / 0.75 / 1.6 | $+5.5$ / 0.85 | 14.8 | – |
| C | DDSA, GLCT (NegBin DPH) | – | $-0.0$ / 1.00 / 0.5 | $+112.6$ / 0.00 | 2.0 | 2 |
| C | DDSA, Boxcar (discretised Gamma) | – | $-0.0$ / 0.90 / 1.0 | $+0.2$ / 0.90 | 13.5 | – |
| C | DSA, Gamma | – | $-0.0$ / 0.95 / 1.0 | $+0.2$ / 0.95 | 13.4 | – |

*Supplement table file: `supp_tables/s5_nonexp.tex`.*

Table S16 gives the results of our runs of the Python code of Di Lauro
et al., with the code unchanged (maximum likelihood; range of the point
estimates over replicates 1–5):

``` julia
tex_table("s6_dsa_python.tex", ["Scenario", "R0 (true 2.25)", "Mean IP (true 9)", "IP SD (true 2.45)", "Shape (true 13.5)", "Fit (s)"])
```

| Scenario | R0 (true 2.25) | Mean IP (true 9) | IP SD (true 2.45) | Shape (true 13.5) | Fit (s) |
|----|----|----|----|----|----|
| A | 2.23–2.33 | 8.9–9.0 | 2.27–2.45 | 13.5–15.4 | 96 |
| B | 3.27–10.60 | 14.4–50.7 | 4.08–13.58 | 7.4–27.7 | 92 |

*Supplement table file: `supp_tables/s6_dsa_python.tex`.*

## The remaining Boxcar $R_0$ bias and the time step

The Boxcar structure recovers the shape of the infectious period. Its
remaining $R_0$ bias halves each time we halve the step, which shows
that it comes from the map (M):

``` julia
s5 = CSV.read(joinpath(D3, "s5_main.csv"), DataFrame); dtr = CSV.read(joinpath(D3, "s5_boxcar_dt.csv"), DataFrame)
box = filter(r -> r.scenario == "A" && r.model == "ddsa_box", s5); dsa = filter(r -> r.scenario == "A" && r.model == "dsa", s5)
dsa_med = Dict(r.rep => r.R0_med for r in eachrow(dsa))
mdrow(["Fit (scenario A, 20 replicates)", "R0 bias (%)", "R0 coverage", "R0 median minus that of our continuous-time DSA (mean over replicates)"]); mdrow(fill("---", 4))
for (lab, d) in (("Boxcar, Δt = 1 d", box), ("Boxcar, Δt = 0.5 d", filter(r -> r.dt == 0.5, dtr)),
                 ("Boxcar, Δt = 0.25 d", filter(r -> r.dt == 0.25, dtr)), ("our continuous-time DSA", dsa))
    bc = bias_cov(d, "R0"); ex = lab == "our continuous-time DSA" ? "—" : @sprintf("%+.3f", mean(r.R0_med - dsa_med[r.rep] for r in eachrow(d)))
    mdrow([lab, @sprintf("%+.1f", bc.bias), @sprintf("%.2f", bc.cov), ex])
end
```

| Fit (scenario A, 20 replicates) | R0 bias (%) | R0 coverage | R0 median minus that of our continuous-time DSA (mean over replicates) |
|----|----|----|----|
| Boxcar, Δt = 1 d | +3.8 | 0.00 | +0.092 |
| Boxcar, Δt = 0.5 d | +1.8 | 0.40 | +0.047 |
| Boxcar, Δt = 0.25 d | +0.8 | 0.45 | +0.024 |
| our continuous-time DSA | -0.3 | 0.40 | — |

# S6 Computational cost

In this section we measure the cost of one log-likelihood evaluation,
one gradient and one complete fit for each model, on one thread of an
otherwise idle machine. The ratios compare our implementations, not the
methods in general.

The scripts are in `S3_S6_dsa_comparison/`: `s6_1_benchmarks.jl` runs
the timings, `s6_0_export.jl` exports the datasets for the code of
KhudaBukhsh et al. and Di Lauro et al., `s6_2_sds.R` runs SDS.Epidemic
and `s6_3_dsa_python.py` runs DSA_refactor. The rerun command first
downloads their code with `get_authors_code.sh` if it is missing. That
script installs the Python packages at the versions we used
(`dsa_refactor_requirements.txt`) and the current CRAN versions of the R
packages, and lists any R package whose version differs from the one we
used. We used R 4.6.1 with SDSMCMC 0.1.0, plyr 1.8.9, MASS 7.3-65, smfsb
1.5, abind 1.4-8, ramcmc 0.1.2 and Rcpp 1.1.2 (`sds_r_versions.txt`).
The results are in Table S17 and in the timings given in the text of the
supplement.

``` julia
./run_method_supplement.sh S6
```

Table S17 gives the timings on one thread; the continuous-time DSA rows
are our Julia re-implementations. ESS/s is the minimum bulk effective
sample size per second of fit, and MH the Metropolis sampler of our
re-implementation of the imputation step:

``` julia
tex_table("s6_cost.tex", ["Section", "Model", "n", "K", "Log-lik. (ms)", "Gradient (ms)", "Fit (s)", "ESS/s"])
```

| Section | Model | n | K | Log-lik. (ms) | Gradient (ms) | Fit (s) | ESS/s |
|----|----|----|----|----|----|----|----|
| S3 | DDSA, geometric | 100 | 90 | 0.00267 | 0.00288 | 5.4 | 391 |
| S3 | DSA | 100 | 90 | 0.152 | 0.216 | 17.0 | 101 |
| S4 | DDSA, geometric, $N$ summed | 100 | 90 | 0.00433 | 0.00662 | 1.9 | 818 |
| S4 | DSA, $n$ summed | 100 | 90 | 0.157 | 0.233 | 19.4 | 106 |
| S4 | DSA, $n$ imputed (MH) | 100 | 90 | 0.149 | – | 19.5 | 87 |
| S3 | DDSA, geometric | 1,000 | 809 | 0.00246 | 0.00283 | 0.3 | 4.5e+03 |
| S3 | DSA | 1,000 | 809 | 0.199 | 0.274 | 22.1 | 63.7 |
| S4 | DDSA, geometric, $N$ summed | 1,000 | 809 | 0.0237 | 0.034 | 2.5 | 699 |
| S4 | DSA, $n$ summed | 1,000 | 809 | 0.219 | 0.301 | 23.8 | 65.4 |
| S4 | DSA, $n$ imputed (MH) | 1,000 | 809 | 0.187 | – | 23.3 | 81.5 |
| S3 | DDSA, geometric | 10,000 | 7,891 | 0.00238 | 0.00273 | 0.3 | 5.08e+03 |
| S3 | DSA | 10,000 | 7,891 | 0.643 | 1.02 | 80.3 | 14 |
| S4 | DDSA, geometric, $N$ summed | 10,000 | 7,891 | 0.184 | 0.301 | 20.8 | 80 |
| S4 | DSA, $n$ summed | 10,000 | 7,891 | 0.902 | 1.28 | 97.9 | 14.1 |
| S4 | DSA, $n$ imputed (MH) | 10,000 | 7,891 | 0.731 | – | 80.8 | 27.1 |
| S5 | DDSA, GLCT (NegBin DPH) | 9,950 | 8,540 | 0.0124 | 0.0151 | 3.0 | 396 |
| S5 | DDSA, Boxcar (discretised Gamma) | 9,950 | 8,540 | 0.367 | 0.375 | 50.9 | 42.2 |
| S5 | DSA, Gamma (renewal grid) | 9,950 | 8,540 | 0.4 | 1.17 | 113.1 | 16.1 |

*Supplement table file: `supp_tables/s6_cost.tex`.*

# S7 Stress tests of DDSA (GLCT structure)

Using the GLCT structure of the main text, we change one assumption at a
time to find where the estimates become biased or the intervals
miscalibrated, and whether the usual checks (posterior predictive
$p$-values, $\hat R$ and divergences) reveal it.

The baseline grid (`S7_stress_tests/baseline_grid/`) uses chain-binomial
outbreaks with $n \in \{100, 1{,}000, 10{,}000\}$,
$R_0 \in \{1.3, 2, 3\}$ and $r \in \{1, 4\}$ (the number of geometric
stages of the simulated infectious period), with 100 replicates per
cell, fitted with the NegBin DPH and geometric models. It also includes
a Gillespie grid with Erlang-4 infectious periods rounded to days, and
refits without the final-size term or with a corrected one. The stress
tests (`S7_stress_tests/`) are 30 levels of 50 replicates each around
the baseline ($n = 1{,}000$, $R_0 = 2$, $r = 4$), classified by the
detection rule described below.

``` julia
./run_method_supplement.sh S7
```

## Baseline grid (Table S18), correctly specified model ($r = 4$)

Table S18 shows the baseline grid when the fitted model matches the
simulated infectious period ($r = 4$):

``` julia
tex_table("s7_grid.tex", ["n", "True R0", "R0: bias / coverage", "Mean IP: bias / coverage", "Geometric model, R0: bias / coverage"])
println("\nAll cells, including \$r = 1\$, spread ratios and convergence, are in `S7_stress_tests/baseline_grid/out/summary_cb.md`.")
```

| n | True R0 | R0: bias / coverage | Mean IP: bias / coverage | Geometric model, R0: bias / coverage |
|----|----|----|----|----|
| 100 | 1.3 | $+1.4$ / 0.38 | $-0.2$ / 0.96 | $+7.5$ / 0.54 |
| 100 | 2 | $+1.2$ / 0.59 | $-0.0$ / 0.91 | $+14.7$ / 0.45 |
| 100 | 3 | $+0.9$ / 0.80 | $+0.8$ / 0.95 | $+20.4$ / 0.39 |
| 1,000 | 1.3 | $+1.1$ / 0.28 | $-0.1$ / 0.84 | $+5.2$ / 0.21 |
| 1,000 | 2 | $-0.5$ / 0.51 | $-0.2$ / 0.94 | $+11.4$ / 0.04 |
| 1,000 | 3 | $-0.1$ / 0.67 | $+0.2$ / 0.95 | $+17.7$ / 0.02 |
| 10,000 | 1.3 | $-0.2$ / 0.17 | $+0.2$ / 0.78 | $+4.6$ / 0.03 |
| 10,000 | 2 | $+0.4$ / 0.52 | $-0.1$ / 0.94 | $+12.2$ / 0.00 |
| 10,000 | 3 | $-0.2$ / 0.68 | $-0.0$ / 0.94 | $+16.9$ / 0.00 |

*Supplement table file: `supp_tables/s7_grid.tex`.*

All cells, including $r = 1$, spread ratios and convergence, are in
`S7_stress_tests/baseline_grid/out/summary_cb.md`.

## Is the final-size term the cause of the narrow $R_0$ intervals?

To check this, we refitted the $n = 1{,}000$, $r = 4$ cells of the
baseline grid with three final-size terms: (a) the Binomial term of the
main text, (b) no final-size term, and (c) a Normal term whose variance
includes the extra-binomial variability of a stochastic epidemic. The
biases in this table are relative, as fractions (+0.011 is +1.1%). The
spread ratio is $k$, and “div \> 0” counts fits with at least one
divergent transition:

``` julia
L = readlines(joinpath(D7, "baseline_grid", "out", "final_step2_summary.md"))
i = findfirst(l -> startswith(l, "| Variant"), L); j = findnext(isempty, L, i)
foreach(println, L[i:j-1])
```

| Variant | True R₀ | R₀ coverage | R₀ spread ratio | R₀ bias | β coverage | β spread ratio | β bias | R̂ \> 1.01 | div \> 0 |
|----|----|----|----|----|----|----|----|----|----|
| \(a\) Binomial final size | 1.3 | 0.28 | 6.02 | +0.011 | 0.37 | 3.99 | +0.012 | 0 | 2 |
| \(a\) Binomial final size | 2.0 | 0.51 | 2.71 | -0.005 | 0.50 | 2.61 | -0.003 | 0 | 2 |
| \(a\) Binomial final size | 3.0 | 0.65 | 1.95 | -0.001 | 0.69 | 1.95 | -0.002 | 0 | 5 |
| \(b\) no final-size term | 1.3 | 0.24 | 5.50 | +0.033 | 0.32 | 4.72 | +0.032 | 0 | 3 |
| \(b\) no final-size term | 2.0 | 0.50 | 2.83 | -0.003 | 0.44 | 2.85 | -0.001 | 0 | 4 |
| \(b\) no final-size term | 3.0 | 0.72 | 1.88 | -0.006 | 0.68 | 1.94 | -0.007 | 0 | 12 |
| \(c\) Normal, corrected variance | 1.3 | 0.28 | 5.44 | +0.025 | 0.33 | 4.54 | +0.026 | 0 | 3 |
| \(c\) Normal, corrected variance | 2.0 | 0.51 | 2.76 | -0.005 | 0.48 | 2.74 | -0.003 | 0 | 4 |
| \(c\) Normal, corrected variance | 3.0 | 0.69 | 1.90 | -0.002 | 0.71 | 1.93 | -0.004 | 0 | 4 |

## Stress tests (Table S19)

We classify each level with the following detection rule. A level fails
if the mean relative bias of the $R_0$ median exceeds 5% (*biased*), or
if, with a smaller bias, its coverage is more than 0.15 below the
baseline’s (*miscalibrated*). A level is *detected* if the share of
flagged fits (posterior predictive $p < 0.05$, $\hat R > 1.01$ or a
divergence) exceeds the baseline’s by at least 0.25, and *silent*
otherwise.

The levels of each factor are:

- **Recording resolution:** continuous-time Erlang-4 infectious periods
  with the given mean, rounded to whole days.
- **Mis-recorded 1-d intervals:** the given share of intervals recorded
  as one day.
- **Missing recoveries:** missing completely at random (MCAR), or
  preferentially when the interval is long.
- **Observation window:** cut at the peak or early in the outbreak, with
  the unrecovered cases right-censored, dropped, or conditioned on
  recovery with a truncation term.
- **Ascertainment:** the share of cases recorded, at random, with $N$
  known.
- **Unknown $N$:** $N$ summed out, with $\rho \sim \mathrm{Beta}(1, 10)$
  (fraction scale) or an Exp(1) prior on the seed count $\rho N$ (count
  scale).
- **Time-varying $\beta$:** halved at the peak (day 45), or declining
  linearly to half its initial value.
- **Heterogeneous mixing:** five out-of-phase subpopulations, or
  gamma-distributed susceptibility with the given coefficient of
  variation (CV).
- **Superspreading:** individual infectiousness drawn from a Gamma
  distribution with mean 1 and the given dispersion.
- **Importations:** one imported case a week.
- **Near-critical:** $R_0 = 1.1$, conditioned on a major outbreak.

“Median r” is the median over fits of the posterior median of $r$, and
“Flagged” the share of flagged fits.

``` julia
tex_table("s7_stress.tex", ["Factor", "Level", "R0: bias / cov. / k", "Mean IP: bias / cov.", "Median r", "Flagged", "Class"])
```

| Factor | Level | R0: bias / cov. / k | Mean IP: bias / cov. | Median r | Flagged | Class |
|----|----|----|----|----|----|----|
| Baseline | baseline | $-0.0$ / 0.54 / 3.1 | $-0.1$ / 0.86 | 4 | 0.28 | – |
| Recording resolution | IP 2 d | $+17.0$ / 0.00 / 2.5 | $-5.3$ / 0.26 | 1 | 1.00 | biased, detected |
|  | IP 4 d | $+12.6$ / 0.06 / 2.5 | $-8.3$ / 0.12 | 1 | 1.00 | biased, detected |
|  | IP 6.33 d | $+13.8$ / 0.06 / 3.0 | $-9.1$ / 0.08 | 1 | 1.00 | biased, detected |
|  | IP 12 d | $+10.2$ / 0.12 / 2.6 | $-8.0$ / 0.26 | 1 | 1.00 | biased, detected |
| Mis-recorded 1-d intervals | 1% | $+11.4$ / 0.02 / 2.3 | $-9.4$ / 0.12 | 1 | 1.00 | biased, detected |
|  | 5% | $+10.8$ / 0.04 / 2.2 | $-11.7$ / 0.00 | 1 | 1.00 | biased, detected |
| Missing recoveries | MCAR 30% | $+0.1$ / 0.66 / 2.6 | $+0.1$ / 0.92 | 4 | 0.24 | unbiased |
|  | MCAR 60% | $-0.9$ / 0.56 / 2.8 | $+0.1$ / 0.94 | 4 | 0.14 | unbiased |
|  | long intervals 30% | $-3.4$ / 0.34 / 2.7 | $-8.7$ / 0.00 | 4 | 0.98 | miscalibrated, detected |
| Observation window | peak, censored | $+0.9$ / 0.68 / 2.2 | $+0.4$ / 0.92 | 4 | 0.10 | unbiased |
|  | peak, dropped | $-4.6$ / 0.48 / 2.1 | $-7.9$ / 0.08 | 4 | 0.02 | unbiased |
|  | peak, truncation term | $+1.3$ / 0.74 / 1.9 | $+0.2$ / 0.94 | 4 | 0.08 | unbiased |
|  | early, censored | $+1.4$ / 0.70 / 1.8 | $-0.5$ / 0.98 | 4 | 0.08 | unbiased |
|  | early, dropped | $-5.0$ / 0.60 / 1.9 | $-9.5$ / 0.38 | 4 | 0.04 | biased, silent |
|  | early, truncation term | $+2.2$ / 0.72 / 1.7 | $+1.9$ / 0.92 | 4 | 0.06 | unbiased |
| Ascertainment | 70% | $-17.9$ / 0.00 / 2.0 | $-2.7$ / 0.44 | 4 | 1.00 | biased, detected |
|  | 40% | $-35.8$ / 0.00 / 2.0 | $-2.9$ / 0.54 | 4 | 1.00 | biased, detected |
| Unknown $N$ | fraction-scale seed prior | $-0.4$ / 0.52 / 2.7 | $-0.0$ / 0.94 | 4 | 0.14 | unbiased |
|  | count-scale seed prior | $-0.5$ / 0.60 / 2.5 | $-0.0$ / 1.00 | 4 | 0.08 | unbiased |
| Time-varying $\beta$ | halved at peak | $+0.4$ / 0.40 / 3.8 | $-0.7$ / 0.92 | 4 | 0.58 | unbiased |
|  | linear decline to 50% | $-18.3$ / 0.00 / 6.6 | $-1.3$ / 0.76 | 4 | 0.80 | biased, detected |
| Heterogeneous mixing | 5 out-of-phase subpopulations | $-31.9$ / 0.00 / 6.2 | $+2.0$ / 0.54 | 4 | 0.96 | biased, detected |
|  | susceptibility CV 0.5 | $-8.2$ / 0.08 / 2.7 | $-0.8$ / 0.86 | 4 | 0.70 | biased, detected |
|  | susceptibility CV 1 | $-23.1$ / 0.00 / 3.0 | $-2.8$ / 0.44 | 4 | 1.00 | biased, detected |
| Superspreading | dispersion 0.5 | $-0.6$ / 0.28 / 4.3 | $-0.1$ / 0.94 | 4 | 0.56 | miscalibrated, detected |
|  | dispersion 0.1 | $+2.0$ / 0.26 / 9.2 | $+0.2$ / 0.74 | 4 | 0.96 | miscalibrated, detected |
| Importations | 1 per week, days 7-91 | $-0.1$ / 0.66 / 2.2 | $+0.6$ / 0.84 | 4 | 0.48 | unbiased |
| Near-critical | $R_0$ 1.1, $n=100$ | $+8.8$ / 0.58 / 1.8 | $+0.2$ / 1.00 | 4 | 0.02 | biased, silent |
|  | $R_0$ 1.1, $n=1,000$ | $+4.6$ / 0.28 / 4.5 | $-0.3$ / 0.94 | 4 | 0.44 | miscalibrated, silent |

*Supplement table file: `supp_tables/s7_stress.tex`.*

# S8 Grid posteriors

In the infection-only fits of the attack-rate sweep (S2, Table S8), the
posterior of the mean infectious period can be bimodal, and the NUTS
chains did not move between the modes. We therefore evaluated these
posteriors on exact grids after summing $r$ out.
`s2_7_grid_vs_turing_check.jl` checks that the grid log densities equal
the Turing log-joints, and running `s2_5_tau_sweep_grid.jl` with
`REFINE=2` checks the effect of refining the grid.

``` julia
./run_method_supplement.sh S8
```

![Figure S2. Marginal posterior of the mean infectious period at N =
1,000: exact grid against the pooled 12-chain
MCMC.](S2_identifiability/figs/s5_fig_meanIP_exact_vs_mcmc_N1000.png)

``` julia
println(readlines(joinpath(S2O, "logs", "s2_7_grid_vs_turing_check.log"))[end])
for c in ("2.0", "3.16")
    f = "s2_5_tausweep_grid_N99_R0$(c)_refine2.txt"; L = readlines(joinpath(S2O, f))
    println("N = 99, true R0 = $c, grid steps halved ($f):")
    for (i, l) in enumerate(L)
        m = match(r"^N=\d+ R0_true=\S+ rung (\d)", l)
        m === nothing || println("   ", MODEL[m[1]], ": ", strip(L[i+1]))
    end
end
```

    overall max difference: 1.82e-12 (supplement S8 states within 2e-12)
    N = 99, true R0 = 2.0, grid steps halved (s2_5_tausweep_grid_N99_R02.0_refine2.txt):
       infection-only DDSA: exact: R0 = 2.15 [1.85, 2.58]  P(R0 > 2.6) = 0.023  | meanIP = 7.56 [4.39, 11.51]  | τ = 0.825 [0.744, 0.887] SD(τ) = 0.0372
       map–Poisson: exact: R0 = 4.35 [1.94, 7.45]  P(R0 > 2.6) = 0.822  | meanIP = 18.19 [5.76, 29.56]  | τ = 0.965 [0.779, 0.99] SD(τ) = 0.0575
    N = 99, true R0 = 3.16, grid steps halved (s2_5_tausweep_grid_N99_R03.16_refine2.txt):
       infection-only DDSA: exact: R0 = 3.21 [2.46, 10.31]  P(R0 > 3.76) = 0.244  | meanIP = 6.71 [3.86, 27.56]  | τ = 0.943 [0.88, 0.982] SD(τ) = 0.0264
       map–Poisson: exact: R0 = 10.11 [3.91, 14.61]  P(R0 > 3.76) = 0.979  | meanIP = 22.69 [8.29, 29.69]  | τ = 0.988 [0.964, 0.995] SD(τ) = 0.0116

# S9 Reproducibility

All runs use the shared environment at the root of the repository (Julia
1.11, Turing.jl 0.49; `Manifest.toml`), which `--project=@.` finds from
this folder, and the repository’s phase-type and integer-parameter
modules, which the scripts find by searching the parent folders. A
single script reruns every section, with a time cap on each process, and
writes the tables of the supplement to `supp_tables/`:

``` julia
# from this folder
./run_method_supplement.sh                # S2 S3 S4 S5 S6 S7 S8 TABLES
./run_method_supplement.sh S2 S8          # selected sections
DRY=1 ./run_method_supplement.sh          # print the commands only
python3 -I make_supplement_tables.py      # tables only: supp_tables/*.tex
```
