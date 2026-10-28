# Discrete dynamic survival analysis of epidemic case line lists
Sandra Montes (`@slmontes`)
2026-10-27

# Introduction

This notebook is a guide to the supplementary analyses of our manuscript
(Sections S2–S5 of the supplementary material). For each section, we
describe what we tested and how, list the scripts that produce the
results and the command to rerun them, and then show the results. The
guide shows every analysis of the supplement, in more detail than the
supplement gives; the tables and figures of the supplement keep their
numbers (Table S10, Figure S2, …). The notebook only reads the saved
outputs, so rendering it takes a few minutes. Rerunning all the analyses
takes about a day on 10 threads and is done with
`run_method_supplement.sh` (see Section [S5](#s5-reproducibility)).

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
our runs of their code, unchanged, in the settings of their own examples
(Table S9).

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
| S2 Identifiability and information in the data (including the grid posteriors) | `S2_identifiability/` | `s2_model.jl` (shared), `s2_0`–`s2_10` | `out/*.txt`, `figs/` (`chains/` is written by the reruns and not distributed) |
| S3 Validation against continuous-time DSA: published codes in their own settings (Table S9), WSU (Table S11) | `S3_S6_dsa_comparison/` | `s6_4`–`s6_8` | `out/s6_4_*`–`out/s6_8_*` |
| S3: agreement on simulated outbreaks (Table S10) | `S3_S6_dsa_comparison/` | `s3_core.jl` (shared), `s3_0_checks.jl`, `s3_1_run.jl` | `out/s3_main.csv` |
| S4 Simulation checks of the stated limitations: calibration (Table S12), daily step (Table S13), stage cap (Table S14) | `S7_stress_tests/baseline_grid/`, `S3_S6_dsa_comparison/` | `sim_core.jl` (shared), `sim_run.jl`, `final_step2_*`, `final_step3_*`; `s3_1_run.jl` (Table S13); `s5_nonexp.jl`, `s5_0_checks.jl`, `s5_boxcar_dt.jl`, `s5_negbin_uncapped.py` | `baseline_grid/out/`; `out/s5_*.csv` |
| Tables of S2–S4 | (this folder) | `make_supplement_tables.py` | `supp_tables/*.tex` (shown in this guide) |

Script and folder names keep their original section numbers (`s4_…`,
`s5_…`, `S7_stress_tests/`, and the section names of
`run_method_supplement.sh`).

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
    # cells shown in grey in the supplement (most fits not converged) are marked † here
    clean(c) = strip(replace(replace(c, r"\\textcolor\{gray\}\{(.*)\}\s*$" => s"\1 †"),
                             "\\,/\\," => " / ", "{+}" => "+", "{,}" => ",", "--" => "–", "\\%" => "%", "\$^{\\dagger}\$" => "",
                             "\$^{a}\$" => " (n = 9,950)", "\\quad " => "", "\\," => " "))
    mdrow(header); mdrow(fill("---", length(header)))
    for r in rows
        # a panel or group heading row, \multicolumn{k}{l}{\textit{...}}, is shown in bold across the first cell
        m = match(r"^\\multicolumn\{\d+\}\{l\}\{\\textit\{(.*)\}\}$", r)
        m === nothing ? mdrow(clean.(split(r, "&"))) : mdrow(["**" * clean(m[1]) * "**"; fill("", length(header) - 1)])
    end
    println("\n*Table file: `supp_tables/$file`.*")
end

"Convergence of a set of fits: number with R̂ > 1.01 and with at least one divergent transition."
conv(df) = println("*Convergence: $(nrow(df)) fits; $(count(>(1.01), df.rhat)) with R̂ > 1.01, $(count(>(0), df.div)) with at least one divergent transition (all kept in the summaries).*")

"Relative bias (%) and 95% CrI coverage of the posterior median over replicates, as in the supplement tables."
function bias_cov(df, p)
    t = df[!, "$(p)_true"]; m = df[!, "$(p)_med"]
    (bias = 100 * mean((m .- t) ./ t), cov = mean(df[!, "$(p)_lo"] .<= t .<= df[!, "$(p)_hi"]))
end
mdrow(cells) = println("| ", join(cells, " | "), " |")
fmtint(n) = replace(string(n), r"(\d)(?=(\d{3})+$)" => s"\1,")
fmtN(n) = fmtint(n)
```

# S1 Prior specifications

Supplement S1 (Tables S1–S5) lists the priors of every main-text
analysis. These are the priors defined in the `@model` blocks of the
case-study notebooks, where the integer-valued parameters ($N$, $r$) are
summed out of the likelihood.

# S2 Identifiability and information in the data

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
by MCMC and on numerical grids; `s2_6` gives numerical grid posteriors
of infection-only DDSA with $N$ unknown, with the seed prior on the
count scale or (`PRIOR=fraction`) on the fraction scale; and `s2_7`
checks the grid log densities against the Turing log-joints. `s2_8`
(numerical grid posteriors of map–Poisson, ODE–Poisson and
infection-only DDSA), `s2_9` (full-data DDSA and the per-$N$ figures)
and `s2_10` (the panel figure, Figure S2, from the saved draws), with
their data from `s2_traj_data.jl`, produce Table S8 and Figure S2, the
fitted trajectories at $N = 99$, $1{,}000$ and $10{,}000$.

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
timing term of DDSA. The first four rows cross Multinomial timing or
Poisson daily counts with the Binomial final-size term or none. The
supplement shows the first and fourth: by the factorisation of Poisson
daily counts into the Multinomial timing term and Poisson$(K; N\tau)$
(Supplement S2), the second row is map–Poisson written as Mult +
Poisson$(K; N\tau)$, and the third scores the total twice (in the
Poisson counts and in the Binomial term), so it is a diagnostic rather
than a model of the data. The never-infected-cell rows add one count
cell for the individuals never infected. The last rows sum $N$ out, with
$\rho \sim \mathrm{Beta}(1, 98)$ on the fraction of initial infectives
(the fraction scale) or with an Exp(1) prior on their number
$I_0 = \rho N$ (the count scale); the NUTS fit with
$N \sim \mathrm{U}(91, 20{,}090)$ on the fraction scale did not converge
(next output), so the supplement reports the numerical grid posteriors
for both seed priors.

``` julia
mdrow(["Likelihood (data of infection-only DDSA)", "R0", "Source"]); mdrow(fill("---", 3))
for l in lines_starting("s2_1_2x2.txt", "2x2")
    mdrow([withnames(strip(split(split(l, "|")[1], "2x2")[2])), grab(l, "R0"), "`s2_1_2x2.txt`"])
end
for l in lines_starting("s2_2_corroborations_misc.txt", "R0≤5")
    mdrow(["Never-infected cell, " * replace(strip(split(split(l, "|")[1], "R0≤5")[2]), " + cell" => "", "continuous ODE" => "ODE"), grab(l, "R0"), "`s2_2_corroborations_misc.txt`"])
end
UL = readlines(joinpath(S2O, "s2_1_unknownN.txt"))
for (i, l) in enumerate(UL)
    startswith(l, "1N") || continue
    rh = parse(Float64, match(r"Rhat≤([0-9.]+)", UL[findnext(x -> occursin("Rhat≤", x), UL, i)])[1])
    est = rh <= 1.01 ? grab(l, "R0") * " (N " * something(match(r"N=([0-9.]+ \[[0-9., ]+\])", l), ["—"])[1] * ")" :
          "chains split between two modes (R̂ = $rh); the pooled draws, $(grab(l, "R0")), are not a posterior summary"
    mdrow([withnames(strip(split(split(l, "|")[1], "1N")[2])) * " (NUTS)", est, "`s2_1_unknownN.txt`"])
end
for (f, lab) in (("s2_6_unknownN_fractionprior_grid.txt", "fraction scale, ρ ~ Beta(1, 98)"), ("s2_6_unknownN_seedprior_grid.txt", "count scale, I0 = ρN ~ Exp(1)"))
    ex = only(lines_starting(f, "Grid posterior"))
    mdrow(["Infection-only DDSA, N ~ U(91, 20,090), seed prior on the $lab (numerical grid)", grab(ex, "R0"), "`$f`"])
end
```

| Likelihood (data of infection-only DDSA) | R0 | Source |
|----|----|----|
| Mult + Binomial (= infection-only DDSA) | 3.11 \[2.44, 4.44\] | `s2_1_2x2.txt` |
| Mult + Poisson(K;Nτ) (≡ map–Poisson) | 4.19 \[2.52, 4.97\] | `s2_1_2x2.txt` |
| Poisson counts + Binomial | 3.11 \[2.46, 4.52\] | `s2_1_2x2.txt` |
| Poisson counts (= map–Poisson) | 4.15 \[2.42, 4.97\] | `s2_1_2x2.txt` |
| Never-infected cell, discrete map | 3.12 \[2.43, 4.44\] | `s2_2_corroborations_misc.txt` |
| Never-infected cell, ODE | 3.14 \[2.46, 4.46\] | `s2_2_corroborations_misc.txt` |
| Infection-only DDSA, N ~ DiscreteUniform(91, 500) (NUTS) | 4.2 \[2.31, 4.96\] (N 94.0 \[91.0, 107.0\]) | `s2_1_unknownN.txt` |
| Infection-only DDSA, N ~ DiscreteUniform(91, 20090) (NUTS) | chains split between two modes (R̂ = 1.735); the pooled draws, 1.1 \[0.5, 4.93\], are not a posterior summary | `s2_1_unknownN.txt` |
| Infection-only DDSA, N ~ U(91, 20,090), seed prior on the fraction scale, ρ ~ Beta(1, 98) (numerical grid) | 4.175 \[2.225, 4.975\] | `s2_6_unknownN_fractionprior_grid.txt` |
| Infection-only DDSA, N ~ U(91, 20,090), seed prior on the count scale, I0 = ρN ~ Exp(1) (numerical grid) | 4.175 \[2.225, 4.975\] | `s2_6_unknownN_seedprior_grid.txt` |

When $N$ has a wide support and the seed prior is on the fraction scale,
the NUTS chains end up in two modes. The per-chain output shows the
second, local mode (a sub-critical outbreak seeded by about 91
infectives) and its lower log-density; the grid posteriors below give
the mass of the sub-critical region that contains it, with the same
priors (fraction scale) and with the count-scale seed prior, and with
the grid steps halved (`REFINE=2`). IP is the mean infectious period and
$\rho N$ the number of initial infectives:

``` julia
foreach(println, filter(l -> occursin("chain", l), readlines(joinpath(S2O, "s2_1_unknownN.txt")))[6:10])
for f in ("s2_6_unknownN_fractionprior_grid.txt", "s2_6_unknownN_fractionprior_grid_refine2.txt", "s2_6_unknownN_seedprior_grid.txt")
    println(f, ":")
    foreach(l -> println("   ", strip(l)), filter(l -> occursin(r"Seed prior|Posterior mass|Grid posterior|max \|grid", l), readlines(joinpath(S2O, f))))
end
```

        chain 1: R0=4.24 [2.34, 4.96]  IP=9.11 [3.8, 12.44]  τ=0.97 [0.86, 0.99]  N=94.0 [91.0, 107.0]  ρ=0.01 [0.0, 0.02]  ρN=0.98 [0.37, 2.25]
        chain 2: R0=0.52 [0.5, 0.58]  IP=13.9 [11.47, 16.28]  τ=0.01 [0.0, 0.04]  N=9753.0 [2525.98, 19491.18]  ρ=0.01 [0.0, 0.04]  ρN=91.41 [67.32, 116.21]
        chain 3: R0=0.52 [0.5, 0.59]  IP=13.89 [11.59, 16.16]  τ=0.01 [0.0, 0.04]  N=9763.5 [2421.75, 19437.03]  ρ=0.01 [0.0, 0.04]  ρN=91.53 [68.49, 115.93]
        chain 4: R0=4.12 [2.09, 4.96]  IP=8.69 [3.13, 12.11]  τ=0.97 [0.82, 0.99]  N=94.0 [91.0, 114.0]  ρ=0.01 [0.0, 0.02]  ρN=0.91 [0.34, 2.04]
        chain mean log-joint: [-50.7, -79.3, -79.3, -50.7] | Rhat≤1.735 ESS≥48 div=0 | 281.0 s
    s2_6_unknownN_fractionprior_grid.txt:
       Seed prior: fraction scale, ρ ~ Beta(1, 98); N ~ U(91, 20090) summed out
       Grid 10×116×90×40 (r, μ step 0.25, R0 step 0.05, ρ ∈ [1e-6, 0.3] 40 log points) in 9 s; max |grid ll − exact ll_I0| over 2000 random cells = 2.4e-5
       Posterior mass with τ < 0.5 (large-N / small-attack region): 3.497084169431877e-5
       Posterior mass with R0 < 1 (sub-critical; mode B of the NUTS fit lies here): 2.854185154468201e-14
       Grid posterior: R0 = 4.175 [2.225, 4.975] | meanIP = 8.875 [3.375, 12.125] | ρ = 0.009 [0.004, 0.023] | τ = 0.971 [0.849, 0.986]
    s2_6_unknownN_fractionprior_grid_refine2.txt:
       Seed prior: fraction scale, ρ ~ Beta(1, 98); N ~ U(91, 20090) summed out
       Grid 10×232×180×80 (r, μ step 0.125, R0 step 0.025, ρ ∈ [1e-6, 0.3] 80 log points) in 64 s; max |grid ll − exact ll_I0| over 2000 random cells = 2.4e-5
       Posterior mass with τ < 0.5 (large-N / small-attack region): 2.0622704740498062e-5
       Posterior mass with R0 < 1 (sub-critical; mode B of the NUTS fit lies here): 3.1963786458035345e-14
       Grid posterior: R0 = 4.188 [2.238, 4.962] | meanIP = 8.938 [3.438, 12.188] | ρ = 0.01 [0.004, 0.023] | τ = 0.971 [0.846, 0.985]
    s2_6_unknownN_seedprior_grid.txt:
       Seed prior: count scale, I0 = ρN ~ Exp(1); N ~ U(91, 20090) summed out
       Grid 10×116×90×40 (r, μ step 0.25, R0 step 0.05, ρ ∈ [1e-6, 0.3] 40 log points) in 9 s; max |grid ll − exact ll_I0| over 2000 random cells = 2.4e-5
       Posterior mass with τ < 0.5 (large-N / small-attack region): 4.50085261429257e-5
       Posterior mass with R0 < 1 (sub-critical; mode B of the NUTS fit lies here): 3.947327604370254e-23
       Grid posterior: R0 = 4.175 [2.225, 4.975] | meanIP = 8.875 [3.375, 12.375] | ρ = 0.009 [0.003, 0.023] | τ = 0.971 [0.853, 0.986]

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

## The attack-rate sweep

We vary the attack rate and compare infection-only DDSA against
map–Poisson, using numerical grid posteriors. The supplement summarises
this table in one paragraph.

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

## Numerical grid posteriors: checks

After the integer-valued parameters are summed out, the models without
recovery intervals have three real-valued parameters, so where NUTS
mixed poorly their posteriors are evaluated on grids. Integration on the
grid is a midpoint rule in which each cell is weighted by its width,
including the unequal steps of the mean infectious period. In the
infection-only fits of the attack-rate sweep at $N = 1{,}000$, the
posterior of the mean infectious period can be bimodal, and the NUTS
chains did not move between the modes (figure below).
`s2_7_grid_vs_turing_check.jl` checks that the grid log densities equal
the Turing log-joints, and running `s2_5_tau_sweep_grid.jl` with
`REFINE=2` checks the effect of halving the grid steps.

![Marginal posterior of the mean infectious period in the attack-rate
sweep at N = 1,000: numerical grid posterior against the pooled 12-chain
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

## Table S8 and Figure S2: larger populations

Table S6 uses a single outbreak with $N = 99$, in which almost everyone
is infected. Here we fit four models to outbreaks with $N = 99$,
$1{,}000$ and $10{,}000$ and compare their fitted trajectories (expected
new infections per day and number infectious) with the true mean-field
trajectory: map–Poisson, ODE–Poisson, infection-only DDSA and full-data
DDSA. Given the same expected daily infections, Poisson counts
conditional on their total follow the Multinomial of the DDSA
infection-time term, so map–Poisson and infection-only DDSA differ only
in how the total number of cases is scored (Poisson($N\tau$) or
Binomial($N$, $\tau$)), and full-data DDSA adds the recovery intervals;
ODE–Poisson replaces the discrete map by an Erlang-$r$ ODE. The $N = 99$
data are those of Table S6. The $N = 1{,}000$ and $10{,}000$ data are
drawn from the discrete map at $R_0 = 3.16$, as in the attack-rate sweep
(at $N = 1{,}000$, the infection days of its $R_0 = 3.16$ cell):
$K \sim \mathrm{Binomial}(N, \tau)$, infection days from the map’s
incidence, and one recovery interval from the simulator’s DPH for each
case and for the initial infective. The data therefore have independent
cases, so these comparisons concern identification, not interval
calibration. Each $N$ is one dataset: the results illustrate the models’
behaviour rather than measure average performance. The models are those
of Table S6, with three changes: $R_0 \sim \mathrm{U}(0.5, 20)$;
$\rho \sim \mathrm{Beta}(1, N-1)$ with the $N$ of each dataset; and the
ODE sums $r$ over $1, \ldots, 40$ without the constraint $\mu > r$,
which only the discrete map needs ($p = r/\mu < 1$). The data are
generated by `s2_traj_data.jl`, which both scripts below include.

Full-data DDSA is fitted with NUTS (4 chains × 4,000 draws, starts from
the prior; `s2_9_trajectories.jl`). Without recovery intervals, the NUTS
fits we ran at $N \ge 1{,}000$ did not reach $\hat R \le 1.01$ (for
example, map–Poisson in the attack-rate sweep at $N = 1{,}000$ and
$R_0 = 3.16$, with 12 chains: $\hat R$ 1.27,
`s2_4_tausweep_mcmc_N1000_12ch_R03.16.txt`). The posteriors of the other
three models are therefore numerical grid posteriors
(`s2_8_grid_poisson.jl`). After $r$ is summed out, the mean infectious
period $\mu$ is gridded (0.025-d steps below 15 d, 0.25 d above). For
each $(\mu, r)$, the conditional posterior of $(R_0, \log\rho)$ is
integrated on its own $49 \times 49$ window, centred by Newton steps on
the conditional mode and spanning ±7 conditional standard deviations.
4,000 independent draws from each grid posterior give the trajectory
bands. `s2_9_trajectories.jl` also draws the figures.

``` julia
# from this folder
ONLY="S2_8 S2_9" ./run_method_supplement.sh S2
./run_method_supplement.sh FIGURES          # s2_10: the panel figure (Figure S2) from the saved draws
```

In the table, the three grid models show the quantiles of their grid
posteriors, and full-data DDSA its pooled NUTS draws. Peak number
infectious is the largest expected number infectious along each
trajectory, summarised over the draws used for the bands. “Line list” is
the peak counted from the simulated line list. The ODE’s mean infectious
period is a continuous-time mean; “recorded” is the mean interval its
draws would give when recorded by day as the data are,
$\mu + E[(1 - W)^+]$ for an infection time uniform within its day (a
recovery on the day of infection is recorded as one day), which is
comparable with the data’s 6.33 d.

The data at $N = 1{,}000$ and $10{,}000$ are generated by the discrete
map, so ODE–Poisson is misspecified for them: to reproduce the incidence
of the map it needs a longer infectious period, and its posterior of $r$
is still rising at $r = 40$. Its rows compare the trends of the two
count models, not the accuracy of the two time scales.

``` julia
"`median [lo, hi]` after `key ` (no `=`) in a line, rounded to integers."
function qint(line, key)
    m = match(Regex(key * raw" ([-0-9.]+) \[([-0-9.]+), ([-0-9.]+)\]"), line)
    m === nothing && return "—"
    f(x) = fmtint(round(Int, parse(Float64, x)))
    "$(f(m[1])) [$(f(m[2])), $(f(m[3]))]"
end
"The line of `L` that starts with `prefix`, and the line after it."
function with_next(L, prefix)
    i = findfirst(l -> startswith(strip(l), prefix), L)
    L[i], L[i+1]
end
grids = (("map", "Map–Poisson", "s2_8_grid_map_poisson_N"), ("ode", "ODE–Poisson", "s2_8_grid_ode_poisson_N"),
         ("inf", "Infection-only DDSA", "s2_8_grid_infonly_N"))
mdrow(["N", "Model", "Posterior", "R0", "Mean IP (d)", "ρN", "Peak number infectious"]); mdrow(fill("---", 7))
for N in (99, 1000, 10000)
    L = readlines(joinpath(S2O, "s2_9_trajectories_N$N.txt"))
    tr = only(filter(l -> startswith(strip(l), "truth:"), L)); re = only(filter(l -> startswith(strip(l), "realised:"), L))
    pk = fmtint(round(Int, parse(Float64, match(r"number infectious peak ([0-9.]+)", tr)[1])))
    mdrow([fmtint(N), "Truth", "—", match(r"R0=([0-9.]+)", tr)[1], match(r"E\[W\]=([0-9.]+)", tr)[1],
           match(r"ρN=([0-9.]+)", L[1])[1], "$pk (line list $(fmtint(parse(Int, match(r"number infectious peak (\d+)", re)[1]))))"])
    for (m, name, stem) in grids
        G = readlines(joinpath(S2O, "$stem$N.txt"))
        ex = only(filter(l -> startswith(strip(l), "posterior:"), G))
        ip = grab(ex, "meanIP")
        if m == "ode"
            rec = only(filter(l -> occursin("recorded mean interval", l), G))
            ip *= " (recorded: " * something(match(r"\]: ([-0-9.]+ \[[-0-9.]+, [-0-9.]+\])", rec), ["—"])[1] * ")"
        end
        _, band = with_next(L, name)
        mdrow(["", name, "numerical grid", grab(ex, "R0"), ip, grab(ex, "ρN"), qint(band, "number infectious peak")])
    end
    fd, band = with_next(L, "Full-data DDSA:")
    mdrow(["", "Full-data DDSA", "NUTS, R̂ ≤ $(match(r"R̂≤([0-9.]+)", fd)[1]), $(match(r"div=(\d+)", fd)[1]) divergences",
           grab(fd, "R0"), grab(fd, "meanIP"), grab(fd, "ρN"), qint(band, "number infectious peak")])
end
println("\n*Source: `S2_identifiability/out/s2_8_grid_*_N*.txt` (grid models), `s2_9_trajectories_N*.txt` (full-data DDSA, peaks, truth).*")
```

| N | Model | Posterior | R0 | Mean IP (d) | ρN | Peak number infectious |
|----|----|----|----|----|----|----|
| 99 | Truth | — | 3.164 | 6.328 | 0.99 | 49 (line list 52) |
|  | Map–Poisson | numerical grid | 10.16 \[3.93, 14.66\] | 22.62 \[8.26, 29.62\] | 1.01 \[0.44, 2.27\] | 91 \[61, 97\] |
|  | ODE–Poisson | numerical grid | 8.84 \[2.98, 12.7\] | 22.38 \[6.99, 29.62\] (recorded: 22.38 \[6.99, 29.62\]) | 0.86 \[0.33, 2.0\] | 93 \[52, 98\] |
|  | Infection-only DDSA | numerical grid | 3.2 \[2.46, 10.28\] | 6.69 \[3.86, 27.38\] | 1.06 \[0.37, 3.15\] | 51 \[31, 93\] |
|  | Full-data DDSA | NUTS, R̂ ≤ 1.001, 0 divergences | 3.13 \[2.75, 3.58\] | 6.4 \[6.04, 6.83\] | 0.97 \[0.5, 1.73\] | 49 \[42, 56\] |
| 1,000 | Truth | — | 3.164 | 6.328 | 1.0 | 495 (line list 475) |
|  | Map–Poisson | numerical grid | 4.6 \[2.72, 12.46\] | 10.69 \[4.86, 28.88\] | 1.7 \[0.9, 2.83\] | 631 \[360, 932\] |
|  | ODE–Poisson | numerical grid | 5.12 \[2.75, 10.6\] | 14.04 \[6.71, 29.38\] (recorded: 14.09 \[6.76, 29.38\]) | 1.48 \[0.85, 2.27\] | 789 \[473, 976\] |
|  | Infection-only DDSA | numerical grid | 3.15 \[2.93, 3.42\] | 6.44 \[4.91, 7.74\] | 1.35 \[0.8, 2.15\] | 458 \[336, 546\] |
|  | Full-data DDSA | NUTS, R̂ ≤ 1.001, 0 divergences | 3.05 \[2.93, 3.17\] | 6.44 \[6.32, 6.57\] | 1.4 \[1.07, 1.81\] | 475 \[454, 496\] |
| 10,000 | Truth | — | 3.164 | 6.328 | 1.0 | 4,928 (line list 4,971) |
|  | Map–Poisson | numerical grid | 3.29 \[2.95, 3.87\] | 6.66 \[5.61, 8.24\] | 1.0 \[0.77, 1.25\] | 5,156 \[4,429, 6,104\] |
|  | ODE–Poisson | numerical grid | 3.27 \[2.94, 3.93\] | 8.04 \[7.04, 9.91\] (recorded: 8.01 \[7.04, 9.99\]) | 0.7 \[0.58, 0.85\] | 6,041 \[5,387, 7,113\] |
|  | Infection-only DDSA | numerical grid | 3.18 \[3.1, 3.25\] | 6.34 \[5.91, 6.69\] | 0.97 \[0.76, 1.17\] | 4,942 \[4,599, 5,235\] |
|  | Full-data DDSA | NUTS, R̂ ≤ 1.001, 0 divergences | 3.17 \[3.13, 3.21\] | 6.33 \[6.29, 6.37\] | 0.97 \[0.86, 1.09\] | 4,936 \[4,865, 5,008\] |

*Source: `S2_identifiability/out/s2_8_grid_*_N*.txt` (grid models),
`s2_9_trajectories_N*.txt` (full-data DDSA, peaks, truth).*

![Figure S2: proportion infectious, $I(t)/N$, for one simulated outbreak
at each population size (columns) under the four models (rows). Coloured
lines and bands: posterior median and 95% CrI of the expected proportion
infectious (parameter uncertainty only); black line: true mean-field
trajectory; grey points: proportion infectious counted from the line
list. Map–Poisson, ODE–Poisson and infection-only DDSA are numerical
grid posteriors; full-data DDSA is fitted with NUTS. ODE–Poisson is
misspecified for these data, which were generated by the discrete map.
Drawn by `s2_10_trajectory_panels.jl` from the saved draws (plotted
values in
`out/s2_10_prevalence_bands.csv`).](S2_identifiability/figs/s2_fig_trajectory_panels.png)

The fitted trajectories at each $N$ with the expected new infections per
day as well. Top: expected new infections per day, with the observed
daily counts (grey bars). Bottom: number infectious, with the number
counted from the line list (grey points).

![Fitted trajectories at $N = 99$ (the data of Table
S6).](S2_identifiability/figs/s2_fig_trajectories_N99.png)

![Fitted trajectories at
$N = 1{,}000$.](S2_identifiability/figs/s2_fig_trajectories_N1000.png)

![Fitted trajectories at
$N = 10{,}000$.](S2_identifiability/figs/s2_fig_trajectories_N10000.png)

The output below gives the checks and diagnostics of each grid
posterior. “check” compares the grid log density with the Turing
log-joint of the same model at 25 random points. “edge share” is the
share of a window’s posterior mass in its outer cells. “Fallback
windows” are built from the coarse search where the Newton steps stopped
before reaching their tolerance (a mode at a prior bound, or, for the
ODE, a line search limited by the numerical noise of the ODE solver near
the mode). “Second-mode coarse cells” are coarse-search cells outside a
window within 10 log units of the coarse maximum. $P(r)$ is the
posterior of $r$ on $1, \ldots, 10$ (map) or $1, \ldots, 40$ (ODE). The
last lines repeat the map–Poisson grid at $N = 10{,}000$ with the $\mu$
steps halved and $98 \times 98$ windows, and the ODE–Poisson grid at
$N = 10{,}000$ with the $\mu$ steps halved over the range that holds its
posterior mass.

``` julia
for (m, name, stem) in grids, N in (99, 1000, 10000)
    println(name, ", N = $N:")
    for p in ("check:", "grid:", "diagnostics:", "P(r)")
        foreach(l -> println("   ", strip(l)), lines_starting("$stem$N.txt", p))
    end
end
for f in filter(f -> occursin(r"^s2_8_grid_(map|ode)_poisson_N10000_refine2", f), readdir(S2O))
    println(f, ":")
    foreach(l -> println("   ", strip(l)), filter(l -> occursin(r"^\s*(posterior|grid):", l), readlines(joinpath(S2O, f))))
end
```

    Map–Poisson, N = 99:
       check: 25 random points, max |Turing log-joint − grid log density| = 6.82e-13
       grid: 620 μ × 10 r, windows 49 × 49 (±7 sd), 4400 finite (μ, r) cells, 282 s on 6 threads
       diagnostics: max window edge share 1.9e-04 (mass-weighted mean 1.2e-06) | fallback windows 73 (posterior mass 1.1e-02) | windows with second-mode coarse cells 0 (posterior mass 0.0e+00)
       P(r) = [0.058, 0.078, 0.093, 0.102, 0.107, 0.111, 0.113, 0.114, 0.113, 0.112]
    Map–Poisson, N = 1000:
       check: 25 random points, max |Turing log-joint − grid log density| = 9.09e-13
       grid: 620 μ × 10 r, windows 49 × 49 (±7 sd), 4400 finite (μ, r) cells, 370 s on 6 threads
       diagnostics: max window edge share 3.6e-10 (mass-weighted mean 2.2e-10) | fallback windows 122 (posterior mass 3.3e-02) | windows with second-mode coarse cells 0 (posterior mass 0.0e+00)
       P(r) = [0.21, 0.228, 0.202, 0.133, 0.082, 0.052, 0.035, 0.025, 0.019, 0.015]
    Map–Poisson, N = 10000:
       check: 25 random points, max |Turing log-joint − grid log density| = 2.91e-11
       grid: 620 μ × 10 r, windows 49 × 49 (±7 sd), 4400 finite (μ, r) cells, 432 s on 6 threads
       diagnostics: max window edge share 2.3e-03 (mass-weighted mean 1.6e-10) | fallback windows 176 (posterior mass 1.3e-02) | windows with second-mode coarse cells 1 (posterior mass 0.0e+00)
       P(r) = [0.0, 0.0, 0.113, 0.671, 0.194, 0.02, 0.001, 0.0, 0.0, 0.0]
    ODE–Poisson, N = 99:
       check: 25 random points, max |Turing log-joint − grid log density| = 5.73e-08
       grid: 620 μ × 40 r, windows 49 × 49 (±7 sd), 24800 finite (μ, r) cells, 8905 s on 10 threads
       diagnostics: max window edge share 1.1e-01 (mass-weighted mean 7.5e-09) | fallback windows 777 (posterior mass 3.4e-02) | windows with second-mode coarse cells 0 (posterior mass 0.0e+00)
       P(r) = [0.008, 0.013, 0.016, 0.018, 0.02, 0.021, 0.022, 0.023, 0.024, 0.024, 0.025, 0.025, 0.025, 0.026, 0.026, 0.026, 0.026, 0.026, 0.027, 0.027, 0.027, 0.027, 0.027, 0.027, 0.027, 0.027, 0.027, 0.027, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028]
    ODE–Poisson, N = 1000:
       check: 25 random points, max |Turing log-joint − grid log density| = 1.29e-08
       grid: 620 μ × 40 r, windows 49 × 49 (±7 sd), 24800 finite (μ, r) cells, 6866 s on 10 threads
       diagnostics: max window edge share 2.6e-01 (mass-weighted mean 1.7e-10) | fallback windows 885 (posterior mass 4.1e-02) | windows with second-mode coarse cells 27 (posterior mass 4.3e-151)
       P(r) = [0.011, 0.016, 0.019, 0.023, 0.027, 0.03, 0.031, 0.032, 0.032, 0.031, 0.031, 0.03, 0.03, 0.029, 0.029, 0.028, 0.028, 0.027, 0.027, 0.026, 0.026, 0.025, 0.025, 0.025, 0.024, 0.024, 0.024, 0.024, 0.023, 0.023, 0.023, 0.023, 0.022, 0.022, 0.022, 0.022, 0.022, 0.022, 0.022, 0.021]
    ODE–Poisson, N = 10000:
       check: 25 random points, max |Turing log-joint − grid log density| = 5.79e-07
       grid: 620 μ × 40 r, windows 49 × 49 (±7 sd), 24800 finite (μ, r) cells, 6777 s on 10 threads
       diagnostics: max window edge share 1.0e+00 (mass-weighted mean 1.0e-10) | fallback windows 914 (posterior mass 3.7e-02) | windows with second-mode coarse cells 944 (posterior mass 4.4e-281)
       P(r) = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.011, 0.014, 0.017, 0.02, 0.024, 0.027, 0.031, 0.034, 0.038, 0.041, 0.044, 0.048, 0.051, 0.054, 0.056, 0.059, 0.062, 0.064, 0.067, 0.069, 0.071, 0.073]
    Infection-only DDSA, N = 99:
       check: 25 random points, max |Turing log-joint − grid log density| = 4.55e-13
       grid: 620 μ × 10 r, windows 49 × 49 (±7 sd), 4400 finite (μ, r) cells, 340 s on 6 threads
       diagnostics: max window edge share 8.8e-08 (mass-weighted mean 2.4e-08) | fallback windows 106 (posterior mass 1.7e-02) | windows with second-mode coarse cells 0 (posterior mass 0.0e+00)
       P(r) = [0.027, 0.082, 0.144, 0.189, 0.194, 0.143, 0.095, 0.06, 0.038, 0.028]
    Infection-only DDSA, N = 1000:
       check: 25 random points, max |Turing log-joint − grid log density| = 9.09e-13
       grid: 620 μ × 10 r, windows 49 × 49 (±7 sd), 4400 finite (μ, r) cells, 407 s on 6 threads
       diagnostics: max window edge share 2.8e-10 (mass-weighted mean 2.5e-10) | fallback windows 165 (posterior mass 3.0e-02) | windows with second-mode coarse cells 0 (posterior mass 0.0e+00)
       P(r) = [0.031, 0.334, 0.321, 0.175, 0.082, 0.039, 0.016, 0.002, 0.0, 0.0]
    Infection-only DDSA, N = 10000:
       check: 25 random points, max |Turing log-joint − grid log density| = 2.91e-11
       grid: 620 μ × 10 r, windows 49 × 49 (±7 sd), 4400 finite (μ, r) cells, 406 s on 6 threads
       diagnostics: max window edge share 2.6e-04 (mass-weighted mean 4.5e-10) | fallback windows 193 (posterior mass 1.6e-02) | windows with second-mode coarse cells 17 (posterior mass 0.0e+00)
       P(r) = [0.0, 0.0, 0.146, 0.733, 0.115, 0.006, 0.0, 0.0, 0.0, 0.0]
    s2_8_grid_map_poisson_N10000_refine2.txt:
       grid: 1240 μ × 10 r, windows 98 × 98 (±7 sd), 8800 finite (μ, r) cells, 611 s on 10 threads
       posterior: R0 = 3.29 [2.95, 3.87] | meanIP = 6.66 [5.62, 8.24] d | ρN = 1.0 [0.77, 1.25] | P(meanIP < 5) = 0.0 | P(meanIP > 20) = 0.0 | P(R0 > 10) = 0.0
    s2_8_grid_ode_poisson_N10000_refine2_win1_mu6.0-13.0.txt:
       grid: 560 μ × 40 r, windows 49 × 49 (±7 sd), 22400 finite (μ, r) cells, 5254 s on 10 threads
       posterior: R0 = 3.27 [2.94, 3.91] | meanIP = 8.03 [7.03, 9.86] d | ρN = 0.7 [0.58, 0.85] | P(meanIP < 5) = 0.0 | P(meanIP > 20) = 0.0 | P(R0 > 10) = 0.0

# S3 Validation against continuous-time DSA

Continuous-time DSA is the established method, so we validate DDSA
against it in three steps: our continuous-time implementations against
the published codes in the settings of the authors’ own examples (Table
S9), DDSA against these implementations on simulated outbreaks (Table
S10), and the case studies against the published analyses (Table S11 for
WSU). Further detail of these analyses follows the tables.

## Table S9: published codes in their own settings

We ran the published SDS R code of KhudaBukhsh et al. in the settings of
its example scripts: Example 2 (10 outbreaks from their Sellke
simulator, $n = 1{,}000$, $\beta = 1.1$, $\gamma = 0.8$, $\rho = 0.05$,
$T = 30$ d, fitted with $n$ known and with $n$ imputed) and example 1
(WSU, $T = 104$, $n$ imputed), with the Gamma(0.001, 0.001) priors of
the examples and four chains of 10,000 iterations (2,000 burn-in). Our
implementations were fitted to the same data with the same priors: our
SDS likelihood with NUTS ($n$ known) and our re-implementation of
Algorithm 4.1 ($n$ imputed), and, for comparison, with the priors of
Section S3. The script also evaluates both log-likelihoods at 20
parameter points per dataset. For the Python code of Di Lauro et al., we
generated five outbreaks with the generator of their synthetic study
($N = 10{,}000$, 50 initial infectives, Gamma infectious period with
mean 9 d and variance 6, $T_f = 80$ d), fitted their likelihood of
infection times $\ell_I$ by maximum likelihood with their particle swarm
and bounds on a 0.1 d grid (the precision of their documented example;
their study script uses 10,000 grid points, which takes about 20 h per
fit), maximised the same $\ell_I$ with our renewal scheme (with their
initial condition, the initial infectives’ ages uniform on 0–5 d) by
Newton steps from their optimum, and evaluated their $\ell_I$ at both
optima on both grids. Finally, our continuous-time implementation was
fitted to the same outbreaks with infection times and the never-infected
count, $n = 9{,}950$ known.

The scripts are in `S3_S6_dsa_comparison/`: `s6_4_sds_own_settings.R`
(their R code), `s6_5_sds_own_settings.jl` (ours),
`s6_6_dsa_own_settings.py` (their Python code; modes `data`, `ll`, `fit`
and `at`) and `s6_7_dsa_own_settings.jl` (ours). They read the published
code from `external/` (`get_authors_code.sh`; `EXT=<folder>` reads it
from elsewhere).

``` julia
# from this folder; PY = the Python of external/.venv
Rscript S3_S6_dsa_comparison/s6_4_sds_own_settings.R 6
julia --project=@. -t 4 S3_S6_dsa_comparison/s6_5_sds_own_settings.jl
for s in 1 2 3 4 5; do $PY -I S3_S6_dsa_comparison/s6_6_dsa_own_settings.py data $s; $PY -I S3_S6_dsa_comparison/s6_6_dsa_own_settings.py ll $s 800; done
$PY -I S3_S6_dsa_comparison/s6_6_dsa_own_settings.py ll 1 10000
for s in 1 2 3 4 5; do $PY -I S3_S6_dsa_comparison/s6_6_dsa_own_settings.py fit $s 800; done > S3_S6_dsa_comparison/out/s6_6_dsa_own_fits.jsonl
julia --project=@. -t 5 S3_S6_dsa_comparison/s6_7_dsa_own_settings.jl
for s in 1 2 3 4 5; do for g in 800 10000; do $PY -I S3_S6_dsa_comparison/s6_6_dsa_own_settings.py at $s $g; done; done > S3_S6_dsa_comparison/out/s6_6_dsa_own_at.jsonl
```

Table S9 summarises the comparison:

``` julia
tex_table("s3_own.tex", ["Setting", "Published code", "Our implementation", "Agreement"])
```

| Setting | Published code | Our implementation | Agreement |
|----|----|----|----|
| SDS, Example 2: 10 outbreaks, $n=1,000$ known | $R_0$ 1.28–1.54 | $R_0$ 1.28–1.54 | $R_0$ medians within 0.003 |
| SDS, Example 2: the same outbreaks, $n$ imputed (Algorithm~4.1) | $R_0$ 1.01–1.81 | $R_0$ 0.98–1.81 | $R_0$ within 0.02 and $N$ within 2% in 8 of 10 |
| SDS, example 1: WSU, $n$ imputed | $R_0$ 1.09 \[1.06, 1.14\], $N$ 7,652 | $R_0$ 1.09 \[1.05, 1.14\], $N$ 7,734 | – |
| DSA, synthetic study: 5 outbreaks, $N=10,000$, infection times only | $\ell_I$ at our optimum | $\ell_I$ at our optimum | within 0.05 log units |

*Table file: `supp_tables/s3_own.tex`.*

The SDS fits outbreak by outbreak (posterior medians of $R_0$ on the
whole-population scale of DDSA, and $\hat R$ of $R_0$; “ours, S3 priors”
uses the priors of Section S3):

``` julia
oR = CSV.read(joinpath(D3, "s6_4_sds_own_R.csv"), DataFrame); oO = CSV.read(joinpath(D3, "s6_5_sds_own_ours.csv"), DataFrame)
getO(set, impl) = only(filter(r -> r.set == set && r.impl == impl, eachrow(oO)))
mdrow(["Dataset", "n", "Published code: R0", "R̂", "N", "Ours, SDS priors: R0", "R̂", "N", "Ours, S3 or S4 priors: R0"]); mdrow(fill("---", 9))
for r in eachrow(sort(oR, [:set, :fitn]))
    a = getO(r.set, r.fitn ? "alg41_sdsprior" : "ours_sdsprior"); b = getO(r.set, r.fitn ? "alg41_s4prior" : "ours_s3prior")
    N(x) = (v = ismissing(x) ? nothing : tryparse(Float64, string(x)); v === nothing || isnan(v) ? "—" : fmtint(round(Int, v)))
    mdrow([r.set, r.fitn ? "imputed" : "known", @sprintf("%.3f", r.R0_med), @sprintf("%.3f", r.rhat_R0), N(r.N_med),
           @sprintf("%.3f", a.R0_med), @sprintf("%.3f", a.rhat), N(a.N_med), @sprintf("%.3f", b.R0_med)])
end
println("\n*Source: `out/s6_4_sds_own_R.csv` (published code), `out/s6_5_sds_own_ours.csv` (ours).*")
```

| Dataset | n | Published code: R0 | R̂ | N | Ours, SDS priors: R0 | R̂ | N | Ours, S3 or S4 priors: R0 |
|----|----|----|----|----|----|----|----|----|
| sellke_rep1 | known | 1.384 | 1.004 | — | 1.385 | 1.001 | — | 1.385 |
| sellke_rep1 | imputed | 1.087 | 1.072 | 1,995 | 1.069 | 1.033 | 2,093 | 1.078 |
| sellke_rep10 | known | 1.492 | 1.007 | — | 1.493 | 1.001 | — | 1.492 |
| sellke_rep10 | imputed | 1.555 | 1.021 | 985 | 1.570 | 1.003 | 975 | 1.574 |
| sellke_rep2 | known | 1.444 | 1.000 | — | 1.447 | 1.002 | — | 1.447 |
| sellke_rep2 | imputed | 1.807 | 1.007 | 770 | 1.811 | 1.002 | 769 | 1.813 |
| sellke_rep3 | known | 1.369 | 1.004 | — | 1.370 | 1.002 | — | 1.370 |
| sellke_rep3 | imputed | 1.319 | 1.022 | 1,127 | 1.332 | 1.005 | 1,106 | 1.334 |
| sellke_rep4 | known | 1.279 | 1.006 | — | 1.279 | 1.002 | — | 1.279 |
| sellke_rep4 | imputed | 1.005 | 1.075 | 2,393 | 0.982 | 1.052 | 2,689 | 1.005 |
| sellke_rep5 | known | 1.402 | 1.007 | — | 1.403 | 1.002 | — | 1.404 |
| sellke_rep5 | imputed | 1.295 | 1.042 | 1,227 | 1.279 | 1.004 | 1,252 | 1.288 |
| sellke_rep6 | known | 1.494 | 1.002 | — | 1.496 | 1.002 | — | 1.496 |
| sellke_rep6 | imputed | 1.688 | 1.024 | 909 | 1.694 | 1.003 | 907 | 1.696 |
| sellke_rep7 | known | 1.491 | 1.002 | — | 1.493 | 1.001 | — | 1.492 |
| sellke_rep7 | imputed | 1.441 | 1.023 | 1,107 | 1.453 | 1.006 | 1,094 | 1.452 |
| sellke_rep8 | known | 1.541 | 1.003 | — | 1.541 | 1.002 | — | 1.540 |
| sellke_rep8 | imputed | 1.742 | 1.013 | 885 | 1.748 | 1.002 | 883 | 1.741 |
| sellke_rep9 | known | 1.395 | 1.003 | — | 1.395 | 1.002 | — | 1.395 |
| sellke_rep9 | imputed | 1.390 | 1.055 | 1,056 | 1.381 | 1.004 | 1,066 | 1.381 |
| wsu | imputed | 1.095 | 1.011 | 7,652 | 1.091 | 1.009 | 7,734 | 1.094 |

*Source: `out/s6_4_sds_own_R.csv` (published code),
`out/s6_5_sds_own_ours.csv` (ours).*

The pointwise check: the largest difference between the two SDS
log-likelihoods over the 20 points of each dataset, with our survival
term at the last infection time (as in their code) or at $T$ (eq. 3.3).
The remaining differences come from their Euler scheme (steps of 0.01 d,
infection times taken at the next grid point):

``` julia
ll = CSV.read(joinpath(D3, "s6_5_sds_own_ll.csv"), DataFrame)
mdrow(["Dataset", "Largest difference, ours (t_last) − theirs", "Range, ours (T) − theirs"]); mdrow(fill("---", 3))
for g in groupby(ll, :set)
    a = g.ll_ours_tlast .- g.ll_sds; b = g.ll_ours_T .- g.ll_sds
    mdrow([g.set[1], @sprintf("%.2f", maximum(abs.(a))), @sprintf("%.2f to %.2f", minimum(b), maximum(b))])
end
```

| Dataset | Largest difference, ours (t_last) − theirs | Range, ours (T) − theirs |
|----|----|----|
| sellke_rep1 | 3.70 | 0.55 to 3.68 |
| sellke_rep2 | 1.90 | -10.69 to 0.34 |
| sellke_rep3 | 2.42 | -1.12 to 2.31 |
| sellke_rep4 | 3.39 | 0.69 to 3.38 |
| sellke_rep5 | 2.33 | -7.93 to 1.23 |
| sellke_rep6 | 1.47 | -4.88 to 0.92 |
| sellke_rep7 | 2.06 | -4.35 to 1.56 |
| sellke_rep8 | 2.22 | -1.91 to 2.04 |
| sellke_rep9 | 1.90 | -10.04 to 0.43 |
| sellke_rep10 | 2.13 | -5.79 to 1.40 |
| wsu | 1.56 | -0.05 to 1.37 |

The DSA comparison outbreak by outbreak. Their optimum is from their
particle swarm on the 0.1 d grid; ours maximises the same $\ell_I$ with
our renewal scheme. Their $\ell_I$ is shown at both optima on the 0.1 d
grid and on the 10,000-point grid of their study. On the fine grid their
$\ell_I$ at our optimum agrees with ours and exceeds their value at
their 0.1 d optimum, so the two optima differ because of the grid. The
last columns are our analysis with the never-infected count ($n$ known):

``` julia
"Flat JSON object on one line (numbers only) as a NamedTuple."
jrow(l) = (; (Symbol(m[1]) => parse(Float64, m[2]) for m in eachmatch(r"\"(\w+)\": ([-0-9.eE+]+)", l))...)
at = [jrow(l) for l in readlines(joinpath(D3, "s6_6_dsa_own_at.jsonl"))]
mle = CSV.read(joinpath(D3, "s6_7_dsa_own_mle.csv"), DataFrame); by = CSV.read(joinpath(D3, "s6_7_dsa_own_bayes.csv"), DataFrame)
mdrow(["Outbreak", "Their optimum: R0", "mean IP", "Our optimum: R0", "mean IP", "ℓ_I ours at ours", "theirs at ours, 10,000 pts", "theirs at theirs, 10,000 pts",
       "With never-infected count: R0", "mean IP"]); mdrow(fill("---", 10))
for r in eachrow(sort(mle, :seed))
    a = only(filter(x -> x.seed == r.seed && x.grid == 10000, at)); b = only(filter(x -> x.seed == r.seed, eachrow(by)))
    mdrow([r.seed, @sprintf("%.2f", r.python_R0), @sprintf("%.1f", r.python_mean), @sprintf("%.2f", r.ours_R0), @sprintf("%.1f", r.ours_mean),
           @sprintf("%.2f", r.ours_llI_at_ours), @sprintf("%.2f", a.llI_python_at_ours), @sprintf("%.2f", a.llI_python_at_python),
           @sprintf("%.3f [%.3f, %.3f]", b.R0_med, b.R0_lo, b.R0_hi), @sprintf("%.2f [%.2f, %.2f]", b.mean_med, b.mean_lo, b.mean_hi)])
end
println("\n*Truth: R0 = 2.25, mean IP 9 d. Sources: `out/s6_6_dsa_own_fits.jsonl`, `out/s6_6_dsa_own_at.jsonl`, `out/s6_7_dsa_own_mle.csv`, `out/s6_7_dsa_own_bayes.csv`; pointwise values in `out/s6_7_dsa_own_ll.csv`.*")
```

| Outbreak | Their optimum: R0 | mean IP | Our optimum: R0 | mean IP | ℓ_I ours at ours | theirs at ours, 10,000 pts | theirs at theirs, 10,000 pts | With never-infected count: R0 | mean IP |
|----|----|----|----|----|----|----|----|----|----|
| 1 | 2.54 | 10.3 | 2.56 | 10.6 | -29862.93 | -29862.92 | -29871.19 | 2.238 \[2.203, 2.275\] | 8.79 \[8.39, 9.13\] |
| 2 | 2.96 | 13.4 | 3.32 | 15.6 | -30216.53 | -30216.49 | -30223.18 | 2.215 \[2.179, 2.251\] | 9.14 \[8.61, 9.54\] |
| 3 | 2.14 | 7.5 | 1.91 | 6.0 | -30270.96 | -30271.01 | -30282.63 | 2.222 \[2.185, 2.258\] | 8.48 \[7.72, 9.29\] |
| 4 | 2.38 | 8.8 | 2.24 | 7.9 | -30172.05 | -30172.04 | -30181.00 | 2.246 \[2.208, 2.286\] | 8.34 \[7.56, 9.14\] |
| 5 | 3.94 | 18.6 | 2.38 | 9.8 | -30093.76 | -30093.77 | -30099.56 | 2.242 \[2.203, 2.279\] | 9.01 \[8.59, 9.36\] |

*Truth: R0 = 2.25, mean IP 9 d. Sources: `out/s6_6_dsa_own_fits.jsonl`,
`out/s6_6_dsa_own_at.jsonl`, `out/s6_7_dsa_own_mle.csv`,
`out/s6_7_dsa_own_bayes.csv`; pointwise values in
`out/s6_7_dsa_own_ll.csv`.*

## Table S10: agreement on simulated outbreaks

We ask whether DDSA, fitted to daily records with the full line list and
$n$ known, estimates $R_0$ and the mean infectious period as well as
continuous-time DSA fitted to exact event times. We simulated exact
event-driven SIR outbreaks with $n \in \{100, 1{,}000, 10{,}000\}$ and
$R_0 \in \{1.5, 2, 4\}$, observed up to near the peak or near the end of
the outbreak, with 50 replicates per cell and an exponential infectious
period with mean 5 d. Each outbreak was analysed with a complete-data
benchmark (the conjugate posterior of $\beta$ and $\gamma$ given the
whole simulated epidemic, a reference for what the data can identify),
with DDSA (geometric infectious period, where GLCT and Boxcar coincide),
and with our implementation of continuous-time DSA, both on exact times
and on daily records spread within the day (each event placed uniformly
within its day). We also fitted DDSA with $\Delta t = 0.5$, 0.25 and 0.1
d.

The scripts are in `S3_S6_dsa_comparison/`: `s3_core.jl` contains the
simulator, observation formats, likelihoods, models and complete-data
benchmark; `s3_0_checks.jl` checks the solver and the likelihoods; and
`s3_1_run.jl` runs the fits.

``` julia
./run_method_supplement.sh S3
```

The tables have one panel for each summary, with the end and peak
cut-offs side by side:

- **Bias:** the difference between the posterior median and the true
  value, relative to the true value, in %, averaged over the replicates.
- **Coverage:** the proportion of replicates whose 95% CrI contains the
  true value.
- **Spread ratio $k$:** the SD of the posterior medians over replicates
  divided by the mean posterior SD. It is close to 1 when the intervals
  match the variability between replicates, and above 1 when they are
  too narrow.
- **DDSA − DSA (exact):** the difference between the two posterior
  medians for the same outbreak, as a percentage of the true $R_0$ (mean
  over replicates, with its 95% CI) and in posterior SDs of DSA (exact)
  (median over replicates). Because both methods see the same outbreak,
  this difference is largely free of how each outbreak departs from the
  mean field, which the complete-data column shows.

The “DSA” columns are our implementation of continuous-time DSA. The
cut-off is where observation ends (near the peak or near the end of the
outbreak), and $n$ is the number initially susceptible, with $m = 0.05n$
initial infectives.

Table S10 of the supplement gives the bias and the paired difference,
with the end and peak cut-offs as panels (a) and (b):

``` julia
tex_table("s3_R0_supp.tex", ["n", "True R0", "Bias %: complete data", "Bias %: DDSA", "Bias %: DSA (exact)", "DDSA − DSA: % of R0 [95% CI]", "DDSA − DSA: posterior SDs"])
```

| n | True R0 | Bias %: complete data | Bias %: DDSA | Bias %: DSA (exact) | DDSA − DSA: % of R0 \[95% CI\] | DDSA − DSA: posterior SDs |
|----|----|----|----|----|----|----|
| **(a) Cut-off near the end** |  |  |  |  |  |  |
| 100 | 1.5 | $-8.8$ | $-10.2$ | $-12.8$ | $+2.6$ $[1.5, 3.6]$ | $0.2$ |
| 100 | 2 | $-11.3$ | $-10.5$ | $-12.7$ | $+2.2$ $[0.7, 3.6]$ | $0.1$ |
| 100 | 4 | $+0.7$ | $+9.3$ | $+0.8$ | $+8.5$ $[6.5, 10.6]$ | $0.8$ |
| 1,000 | 1.5 | $-0.9$ | $-0.9$ | $-1.0$ | $+0.1$ $[-0.1, 0.4]$ | $0.1$ |
| 1,000 | 2 | $+0.6$ | $+0.7$ | $+0.0$ | $+0.7$ $[0.4, 1.0]$ | $0.3$ |
| 1,000 | 4 | $-0.9$ | $+6.1$ | $-1.2$ | $+7.2$ $[6.6, 7.9]$ | $2.4$ |
| 10,000 | 1.5 | $-0.3$ | $-0.2$ | $-0.1$ | $-0.1$ $[-0.2, 0.0]$ | $-0.1$ |
| 10,000 | 2 | $-0.1$ | $+0.7$ | $-0.1$ | $+0.8$ $[0.7, 0.9]$ | $1.1$ |
| 10,000 | 4 | $-0.0$ | $+7.6$ | $-0.0$ | $+7.6$ $[7.4, 7.8]$ | $8.0$ |
| **(b) Cut-off near the peak** |  |  |  |  |  |  |
| 100 | 1.5 | $+0.1$ | $+3.0$ | $-0.2$ | $+3.2$ $[1.7, 4.7]$ | $0.1$ |
| 100 | 2 | $-12.5$ | $-12.3$ | $-14.0$ | $+1.7$ $[0.5, 2.9]$ | $0.0$ |
| 100 | 4 | $+0.9$ | $+10.1$ | $+4.8$ | $+5.3$ $[4.1, 6.5]$ | $0.2$ |
| 1,000 | 1.5 | $-1.4$ | $-2.0$ | $-2.1$ | $+0.1$ $[-0.2, 0.3]$ | $0.0$ |
| 1,000 | 2 | $-2.4$ | $-2.6$ | $-2.7$ | $+0.2$ $[0.0, 0.3]$ | $0.0$ |
| 1,000 | 4 | $-1.2$ | $+0.8$ | $-0.7$ | $+1.4$ $[1.1, 1.7]$ | $0.2$ |
| 10,000 | 1.5 | $-0.5$ | $-0.8$ | $-0.7$ | $-0.1$ $[-0.2, -0.1]$ | $-0.1$ |
| 10,000 | 2 | $-0.1$ | $-0.4$ | $-0.3$ | $-0.1$ $[-0.1, 0.0]$ | $0.0$ |
| 10,000 | 4 | $-0.1$ | $+0.7$ | $-0.1$ | $+0.8$ $[0.7, 0.9]$ | $0.4$ |

*Table file: `supp_tables/s3_R0_supp.tex`.*

The full table, with coverage, the spread ratio $k$ and continuous-time
DSA on daily records spread within the day:

``` julia
s3 = CSV.read(joinpath(D3, "s3_main.csv"), DataFrame)
conv(filter(r -> r.model != "exact", s3))
println()
const S3HEAD = ["n", "True R0", "End: complete data", "End: DDSA (daily)", "End: DSA (exact)", "End: DSA (daily)",
                "Peak: complete data", "Peak: DDSA (daily)", "Peak: DSA (exact)", "Peak: DSA (daily)"]
const DTPHEAD = ["True R0", "Δt = 1 d", "Δt = 0.5 d", "Δt = 0.25 d", "Δt = 0.1 d", "Coverage, DDSA Δt = 1 d", "Coverage, DDSA Δt = 0.1 d", "Coverage, DSA (exact)"]
const DTHEAD = ["True R0", "DDSA, Δt = 1 d", "Δt = 0.5 d", "Δt = 0.25 d", "Δt = 0.1 d", "DSA (exact times)", "Complete data"]
tex_table("s3_R0.tex", S3HEAD)
```

*Convergence: 3150 fits; 7 with R̂ \> 1.01, 106 with at least one
divergent transition (all kept in the summaries).*

| n | True R0 | End: complete data | End: DDSA (daily) | End: DSA (exact) | End: DSA (daily) | Peak: complete data | Peak: DDSA (daily) | Peak: DSA (exact) | Peak: DSA (daily) |
|----|----|----|----|----|----|----|----|----|----|
| **(a) Relative bias of the posterior median (%)** |  |  |  |  |  |  |  |  |  |
| 100 | 1.5 | $-8.8$ | $-10.2$ | $-12.8$ | $-12.8$ | $+0.1$ | $+3.0$ | $-0.2$ | $-0.1$ |
| 100 | 2 | $-11.3$ | $-10.5$ | $-12.7$ | $-12.7$ | $-12.5$ | $-12.3$ | $-14.0$ | $-14.4$ |
| 100 | 4 | $+0.7$ | $+9.3$ | $+0.8$ | $+0.7$ | $+0.9$ | $+10.1$ | $+4.8$ | $+4.3$ |
| 1,000 | 1.5 | $-0.9$ | $-0.9$ | $-1.0$ | $-1.1$ | $-1.4$ | $-2.0$ | $-2.1$ | $-2.4$ |
| 1,000 | 2 | $+0.6$ | $+0.7$ | $+0.0$ | $+0.0$ | $-2.4$ | $-2.6$ | $-2.7$ | $-3.0$ |
| 1,000 | 4 | $-0.9$ | $+6.1$ | $-1.2$ | $-1.6$ | $-1.2$ | $+0.8$ | $-0.7$ | $-1.4$ |
| 10,000 | 1.5 | $-0.3$ | $-0.2$ | $-0.1$ | $-0.1$ | $-0.5$ | $-0.8$ | $-0.7$ | $-1.1$ |
| 10,000 | 2 | $-0.1$ | $+0.7$ | $-0.1$ | $-0.1$ | $-0.1$ | $-0.4$ | $-0.3$ | $-0.6$ |
| 10,000 | 4 | $-0.0$ | $+7.6$ | $-0.0$ | $-0.4$ | $-0.1$ | $+0.7$ | $-0.1$ | $-0.9$ |
| **(b) Coverage of the 95% CrI** |  |  |  |  |  |  |  |  |  |
| 100 | 1.5 | 0.94 | 0.38 | 0.32 | 0.32 | 0.94 | 0.64 | 0.64 | 0.62 |
| 100 | 2 | 0.80 | 0.38 | 0.44 | 0.46 | 0.86 | 0.70 | 0.70 | 0.68 |
| 100 | 4 | 1.00 | 0.82 | 0.88 | 0.90 | 0.94 | 0.90 | 0.92 | 0.88 |
| 1,000 | 1.5 | 0.96 | 0.36 | 0.42 | 0.42 | 0.88 | 0.66 | 0.64 | 0.68 |
| 1,000 | 2 | 0.96 | 0.42 | 0.52 | 0.50 | 0.98 | 0.64 | 0.66 | 0.62 |
| 1,000 | 4 | 0.96 | 0.54 | 0.84 | 0.82 | 1.00 | 0.88 | 0.88 | 0.86 |
| 10,000 | 1.5 | 0.96 | 0.42 | 0.46 | 0.44 | 0.92 | 0.70 | 0.66 | 0.70 |
| 10,000 | 2 | 0.96 | 0.62 | 0.64 | 0.62 | 1.00 | 0.68 | 0.72 | 0.70 |
| 10,000 | 4 | 0.98 | 0.00 | 0.92 | 0.92 | 0.92 | 0.92 | 0.90 | 0.86 |
| **(c) Spread ratio $k$** |  |  |  |  |  |  |  |  |  |
| 100 | 1.5 | 1.4 | 4.5 | 4.4 | 4.4 | 1.1 | 1.9 | 1.9 | 1.9 |
| 100 | 2 | 1.7 | 4.4 | 4.2 | 4.2 | 1.3 | 1.9 | 1.8 | 1.8 |
| 100 | 4 | 0.7 | 1.3 | 1.2 | 1.2 | 0.9 | 1.1 | 1.1 | 1.1 |
| 1,000 | 1.5 | 1.0 | 3.7 | 3.7 | 3.7 | 1.1 | 1.9 | 1.9 | 1.8 |
| 1,000 | 2 | 1.0 | 2.6 | 2.4 | 2.4 | 1.0 | 1.9 | 1.9 | 1.9 |
| 1,000 | 4 | 0.9 | 1.5 | 1.4 | 1.4 | 0.9 | 1.2 | 1.2 | 1.3 |
| 10,000 | 1.5 | 1.1 | 3.7 | 3.6 | 3.6 | 1.2 | 2.0 | 2.0 | 2.0 |
| 10,000 | 2 | 0.9 | 2.2 | 2.2 | 2.2 | 0.9 | 1.7 | 1.7 | 1.7 |
| 10,000 | 4 | 0.8 | 1.4 | 1.3 | 1.3 | 1.0 | 1.2 | 1.2 | 1.2 |

*Table file: `supp_tables/s3_R0.tex`.*

DDSA and our implementation of continuous-time DSA both gave $R_0$
intervals that were too narrow, while the complete-data benchmark was
close to nominal for $n \ge 1{,}000$. Section S4 relates this to the
independence assumption of the likelihood (L). The $R_0$ bias at
$R_0 = 4$ and the +2% bias in the mean infectious period from same-day
removals shrank as the step was refined (Table S13), which points to the
map (M).

## Mean infectious period

The results for the mean infectious period, laid out as Table S10 (bias
and coverage):

``` julia
tex_table("s3_meanIP.tex", S3HEAD)
```

| n | True R0 | End: complete data | End: DDSA (daily) | End: DSA (exact) | End: DSA (daily) | Peak: complete data | Peak: DDSA (daily) | Peak: DSA (exact) | Peak: DSA (daily) |
|----|----|----|----|----|----|----|----|----|----|
| **(a) Relative bias of the posterior median (%)** |  |  |  |  |  |  |  |  |  |
| 100 | 1.5 | $-4.7$ | $-2.5$ | $-6.5$ | $-6.0$ | $+2.3$ | $+8.5$ | $+8.8$ | $+9.8$ |
| 100 | 2 | $-6.6$ | $-2.1$ | $-4.9$ | $-4.4$ | $-5.7$ | $-1.3$ | $+0.4$ | $+0.9$ |
| 100 | 4 | $-0.7$ | $+1.8$ | $+1.2$ | $+1.8$ | $+1.6$ | $-4.6$ | $+9.1$ | $+11.1$ |
| 1,000 | 1.5 | $-0.8$ | $+1.5$ | $-1.3$ | $-0.8$ | $-1.8$ | $-1.3$ | $-1.1$ | $-0.2$ |
| 1,000 | 2 | $+0.4$ | $+3.6$ | $+1.5$ | $+2.0$ | $-1.6$ | $-3.5$ | $-1.2$ | $-0.2$ |
| 1,000 | 4 | $-0.1$ | $+1.5$ | $+0.1$ | $+0.7$ | $-0.2$ | $-11.5$ | $+0.5$ | $+2.3$ |
| 10,000 | 1.5 | $-0.4$ | $+2.1$ | $-0.5$ | $+0.0$ | $-0.2$ | $-0.5$ | $-0.1$ | $+0.6$ |
| 10,000 | 2 | $-0.0$ | $+2.5$ | $+0.1$ | $+0.6$ | $+0.1$ | $-2.5$ | $+0.2$ | $+1.1$ |
| 10,000 | 4 | $+0.0$ | $+1.6$ | $+0.0$ | $+0.8$ | $-0.2$ | $-11.9$ | $-0.1$ | $+1.9$ |
| **(b) Coverage of the 95% CrI** |  |  |  |  |  |  |  |  |  |
| 100 | 1.5 | 0.96 | 0.80 | 0.78 | 0.78 | 0.96 | 0.94 | 0.94 | 0.92 |
| 100 | 2 | 0.96 | 0.90 | 0.84 | 0.86 | 0.98 | 1.00 | 0.96 | 0.98 |
| 100 | 4 | 0.98 | 0.98 | 0.96 | 0.94 | 0.98 | 0.96 | 0.96 | 0.98 |
| 1,000 | 1.5 | 0.94 | 0.84 | 0.82 | 0.82 | 0.90 | 0.90 | 0.92 | 0.92 |
| 1,000 | 2 | 0.94 | 0.74 | 0.84 | 0.80 | 0.96 | 0.94 | 0.98 | 0.96 |
| 1,000 | 4 | 0.98 | 0.90 | 0.96 | 0.94 | 0.94 | 0.48 | 0.94 | 0.96 |
| 10,000 | 1.5 | 0.90 | 0.50 | 0.68 | 0.72 | 0.94 | 0.92 | 0.96 | 0.96 |
| 10,000 | 2 | 0.98 | 0.32 | 0.88 | 0.82 | 1.00 | 0.74 | 1.00 | 0.96 |
| 10,000 | 4 | 0.94 | 0.66 | 0.96 | 0.88 | 0.94 | 0.00 | 0.94 | 0.90 |

*Table file: `supp_tables/s3_meanIP.tex`.*

Its convergence in the step $\Delta t$, laid out as Table S13
($n = 1{,}000$, end cut-off). At the daily step, a recovery on the day
of infection is recorded as a one-day interval, which raises the mean
infectious period by about 2%; the bias shrinks as the records and the
map are refined together:

``` julia
tex_table("s3_dt_meanIP.tex", DTHEAD)
```

| True R0 | DDSA, Δt = 1 d | Δt = 0.5 d | Δt = 0.25 d | Δt = 0.1 d | DSA (exact times) | Complete data |
|----|----|----|----|----|----|----|
| **(a) Relative bias of the posterior median (%)** |  |  |  |  |  |  |
| 1.5 | $+1.5$ | $-0.2$ | $-0.8$ | $-1.1$ | $-1.3$ | $-0.8$ |
| 2 | $+3.6$ | $+2.1$ | $+1.6$ | $+1.4$ | $+1.5$ | $+0.4$ |
| 4 | $+1.5$ | $+0.3$ | $+0.0$ | $-0.0$ | $+0.1$ | $-0.1$ |
| **(b) Coverage of the 95% CrI** |  |  |  |  |  |  |
| 1.5 | 0.84 | 0.88 | 0.86 | 0.86 | 0.82 | 0.94 |
| 2 | 0.74 | 0.80 | 0.82 | 0.84 | 0.84 | 0.94 |
| 4 | 0.90 | 0.98 | 0.98 | 0.98 | 0.96 | 0.98 |

*Table file: `supp_tables/s3_dt_meanIP.tex`.*

## Paired comparison and convergence

DDSA and our implementation of continuous-time DSA analyse the same
outbreaks, so their difference in coverage and in the posterior median
is estimated replicate by replicate (DDSA with $\Delta t = 1$ against
continuous-time DSA on exact times, $R_0$; mean and 95% interval of the
paired differences over the 50 replicates). The second output checks
whether excluding the fits with $\hat R > 1.01$ changes any entry by
more than 0.1 percentage points of bias or 0.01 of coverage. The third
gives the largest change in any entry when the fits with at least one
divergent transition are excluded as well.

``` julia
key(r) = (r.cut, r.n, r.R0, r.rep)
dd = Dict(key(r) => r for r in eachrow(filter(r -> r.model == "ddsa" && r.dt == 1.0, s3)))
de = Dict(key(r) => r for r in eachrow(filter(r -> r.model == "dsa_exact", s3)))
covered(r) = r.R0_lo <= r.R0_true <= r.R0_hi
ci(x) = (m = mean(x); h = 1.96 * std(x) / sqrt(length(x)); @sprintf("%+.2f [%+.2f, %+.2f]", m, m - h, m + h))
mdrow(["Cut-off", "n", "True R0", "Coverage, DDSA − DSA", "Median of R0, DDSA − DSA (% of truth)"]); mdrow(fill("---", 5))
for cut in ("end", "peak"), n in (100, 1000, 10000), R0 in (1.5, 2.0, 4.0)
    ks = [k for k in keys(dd) if k[1:3] == (cut, n, R0) && haskey(de, k)]
    mdrow([cut, fmtN(n), R0, ci([covered(dd[k]) - covered(de[k]) for k in ks]), ci([100 * (dd[k].R0_med - de[k].R0_med) / R0 for k in ks])])
end
println()
fits = filter(r -> r.model != "exact", s3); bad = filter(r -> r.rhat > 1.01, fits)
println("*Fits with R̂ > 1.01: $(nrow(bad)). Entries of the S3 tables that change when they are excluded (bias in percentage points, coverage):*\n")
for g in groupby(fits, [:cut, :n, :R0, :model, :dt])
    any(>(1.01), g.rhat) || continue
    good = filter(r -> r.rhat <= 1.01, g)
    for p in ("R0", "μ")
        a = bias_cov(g, p); b = bias_cov(good, p)
        (abs(a.bias - b.bias) > 0.1 || abs(a.cov - b.cov) > 0.01) &&
            println("- ", join([g.cut[1], g.n[1], g.R0[1], g.model[1], "Δt=$(g.dt[1])", p], ", "), @sprintf(": bias %+.1f → %+.1f, coverage %.2f → %.2f", a.bias, b.bias, a.cov, b.cov))
    end
end
println()
ok(r) = r.rhat <= 1.01 && r.div == 0
worst = Dict(p => [0.0, 0.0, ""] for p in ("R0", "μ"))
for g in groupby(fits, [:cut, :n, :R0, :model, :dt])
    good = filter(ok, g); nrow(good) == nrow(g) && continue
    for p in ("R0", "μ")
        a = bias_cov(g, p); b = bias_cov(good, p)
        abs(a.bias - b.bias) > worst[p][1] && (worst[p][1] = abs(a.bias - b.bias); worst[p][3] = join([g.cut[1], "n = $(fmtN(g.n[1]))", "R0 = $(g.R0[1])", g.model[1]], ", "))
        worst[p][2] = max(worst[p][2], abs(a.cov - b.cov))
    end
end
println("*Excluding also the fits with a divergent transition ($(count(!ok, eachrow(fits))) fits in all), the largest changes in any entry are:*\n")
for (p, lab) in (("R0", "R0"), ("μ", "mean infectious period"))
    println(@sprintf("- %s: bias %.1f percentage points (%s), coverage %.2f", lab, worst[p][1], worst[p][3], worst[p][2]))
end
```

| Cut-off | n | True R0 | Coverage, DDSA − DSA | Median of R0, DDSA − DSA (% of truth) |
|----|----|----|----|----|
| end | 100 | 1.5 | +0.06 \[-0.04, +0.16\] | +2.55 \[+1.47, +3.64\] |
| end | 100 | 2.0 | -0.06 \[-0.13, +0.01\] | +2.15 \[+0.68, +3.62\] |
| end | 100 | 4.0 | -0.06 \[-0.16, +0.04\] | +8.54 \[+6.49, +10.59\] |
| end | 1,000 | 1.5 | -0.06 \[-0.15, +0.03\] | +0.15 \[-0.08, +0.37\] |
| end | 1,000 | 2.0 | -0.10 \[-0.18, -0.02\] | +0.72 \[+0.42, +1.03\] |
| end | 1,000 | 4.0 | -0.30 \[-0.49, -0.11\] | +7.24 \[+6.61, +7.87\] |
| end | 10,000 | 1.5 | -0.04 \[-0.09, +0.01\] | -0.08 \[-0.15, -0.02\] |
| end | 10,000 | 2.0 | -0.02 \[-0.15, +0.11\] | +0.76 \[+0.66, +0.86\] |
| end | 10,000 | 4.0 | -0.92 \[-1.00, -0.84\] | +7.58 \[+7.37, +7.79\] |
| peak | 100 | 1.5 | +0.00 \[+0.00, +0.00\] | +3.16 \[+1.66, +4.66\] |
| peak | 100 | 2.0 | +0.00 \[+0.00, +0.00\] | +1.68 \[+0.51, +2.85\] |
| peak | 100 | 4.0 | -0.02 \[-0.06, +0.02\] | +5.29 \[+4.12, +6.47\] |
| peak | 1,000 | 1.5 | +0.02 \[-0.02, +0.06\] | +0.05 \[-0.16, +0.26\] |
| peak | 1,000 | 2.0 | -0.02 \[-0.09, +0.05\] | +0.16 \[+0.01, +0.30\] |
| peak | 1,000 | 4.0 | +0.00 \[-0.06, +0.06\] | +1.42 \[+1.12, +1.73\] |
| peak | 10,000 | 1.5 | +0.04 \[-0.01, +0.09\] | -0.12 \[-0.18, -0.07\] |
| peak | 10,000 | 2.0 | -0.04 \[-0.09, +0.01\] | -0.06 \[-0.09, -0.02\] |
| peak | 10,000 | 4.0 | +0.02 \[-0.05, +0.09\] | +0.79 \[+0.71, +0.87\] |

*Fits with R̂ \> 1.01: 7. Entries of the S3 tables that change when they
are excluded (bias in percentage points, coverage):*

- end, 1000, 2.0, ddsa, Δt=0.5, R0: bias +0.3 → +0.3, coverage 0.46 →
  0.45
- end, 10000, 2.0, dsa_exact, Δt=0.0, R0: bias -0.1 → -0.1, coverage
  0.64 → 0.65
- peak, 1000, 4.0, dsa_exact, Δt=0.0, R0: bias -0.7 → -0.8, coverage
  0.88 → 0.88
- end, 100, 1.5, dsa_exact, Δt=0.0, R0: bias -12.8 → -11.9, coverage
  0.32 → 0.31
- end, 100, 1.5, dsa_exact, Δt=0.0, μ: bias -6.5 → -5.7, coverage 0.78 →
  0.79
- end, 100, 1.5, dsa_jitter, Δt=1.0, R0: bias -12.8 → -13.4, coverage
  0.32 → 0.31
- end, 100, 1.5, dsa_jitter, Δt=1.0, μ: bias -6.0 → -6.1, coverage 0.78
  → 0.78

*Excluding also the fits with a divergent transition (113 fits in all),
the largest changes in any entry are:*

- R0: bias 4.4 percentage points (end, n = 100, R0 = 2.0, dsa_jitter),
  coverage 0.04
- mean infectious period: bias 2.8 percentage points (end, n = 100, R0 =
  2.0, dsa_jitter), coverage 0.03

## Table S11: WSU, the population-size step

The WSU line list fitted with our continuous-time implementation (exact
times, $T = 104$ d, the 11 cases at $t = 0$ as initial infectives),
changing only the population-size step: the imputation step of Algorithm
4.1 with the priors of the SDS example (which reproduces the published
code, Table S9); the exact conditional draw of $n$ under the complete
Binomial final-size term, $n - K \sim \mathrm{NegBin}(K + 1, q_T)$, with
the same sampler and priors; and $N$ summed out of the complete Binomial
term with NUTS and the priors of Section S3. The complete-Binomial arms
use the bound $N \le 18{,}000$ of the DDSA analysis. The script is
`S3_S6_dsa_comparison/s6_8_wsu_nstep.jl` (about 10 min).

``` julia
julia --project=@. -t 3 S3_S6_dsa_comparison/s6_8_wsu_nstep.jl
```

``` julia
tex_table("s3_wsu_nstep.tex", ["Population-size step", "N", "R0", "Mean IP (d)", "R̂"])
```

| Population-size step | N | R0 | Mean IP (d) | R̂ |
|----|----|----|----|----|
| Imputation step of Algorithm~4.1, $n-K\sim\mathrm{NegBin}(K,\tau_\infty)$, no $\binom{n}{K}$ | 7,786 \[6,731, 9,052\] | 1.09 \[1.05, 1.14\] | 5.13 | 1.008 |
| Exact conditional draw under the complete Binomial, $n-K\sim\mathrm{NegBin}(K+1,q_T)$ | 12,785 \[9,391, 17,603\] | 0.98 \[0.94, 1.04\] | 5.03 | 1.032 |
| $N$ summed out of the complete Binomial (NUTS) | 12,232 \[9,092, 17,124\] | 0.99 \[0.94, 1.05\] | 5.04 | 1.005 |
| DDSA (main text, Section~3(c)) | 14,144 | 0.97 | 5.12 | – |

*Table file: `supp_tables/s3_wsu_nstep.tex`.*

# S4 Simulation checks of the stated limitations

The simulations behind the limitations stated in the main text: interval
calibration (Table S12), the daily step (Table S13) and the GLCT stage
cap (Table S14).

## Table S12: interval calibration

The baseline grid (`S7_stress_tests/baseline_grid/`) uses chain-binomial
outbreaks with $n \in \{100, 1{,}000, 10{,}000\}$,
$R_0 \in \{1.3, 2, 3\}$ and $r \in \{1, 4\}$ (the number of geometric
stages of the simulated infectious period), with 100 replicates per
cell, fitted with the NegBin DPH and geometric models. It also includes
refits without the final-size term or with a corrected one, and fits to
data drawn from the DDSA working model itself, in which cases are
independent.

``` julia
./run_method_supplement.sh S7
```

Table S12 shows the baseline grid when the fitted model matches the
simulated infectious period ($r = 4$), next to the same model fitted to
data drawn from the working model (next subsections) and the bias of the
misspecified geometric model. Panel (a) is $R_0$ and panel (b) the mean
infectious period:

``` julia
tex_table("s7_grid.tex", ["n", "True R0", "Chain binomial, NegBin DPH: bias %", "cov.", "k",
                          "Working-model data, NegBin DPH: bias %", "cov.", "k", "Chain binomial, geometric: bias %"])
```

| n | True R0 | Chain binomial, NegBin DPH: bias % | cov. | k | Working-model data, NegBin DPH: bias % | cov. | k | Chain binomial, geometric: bias % |
|----|----|----|----|----|----|----|----|----|
| **(a) $R_0$** |  |  |  |  |  |  |  |  |
| 100 | 1.3 | $+1.4$ | 0.38 | 3.0 | $+0.3$ | 0.94 | 1.1 | $+7.5$ |
| 100 | 2 | $+1.2$ | 0.59 | 2.4 | $-0.1$ | 0.94 | 1.1 | $+14.7$ |
| 100 | 3 | $+0.9$ | 0.80 | 1.6 | $+1.6$ | 0.97 | 0.9 | $+20.4$ |
| 1,000 | 1.3 | $+1.1$ | 0.28 | 6.0 | $+0.1$ | 0.90 | 1.0 | $+5.2$ |
| 1,000 | 2 | $-0.5$ | 0.51 | 2.7 | $-0.0$ | 0.94 | 1.0 | $+11.4$ |
| 1,000 | 3 | $-0.1$ | 0.67 | 2.0 | $+0.5$ | 0.97 | 1.0 | $+17.7$ |
| 10,000 | 1.3 | $-0.2$ | 0.17 | 6.9 | $+0.0$ | 0.95 | 1.0 | $+4.6$ |
| 10,000 | 2 | $+0.4$ | 0.52 | 2.9 | $-0.0$ | 0.96 | 1.0 | $+12.2$ |
| 10,000 | 3 | $-0.2$ | 0.68 | 1.9 | $-0.0$ | 0.95 | 1.0 | $+16.9$ |
| **(b) Mean infectious period** |  |  |  |  |  |  |  |  |
| 100 | 1.3 | $-0.2$ | 0.96 | 1.1 | $-0.2$ | 0.94 | 1.1 | $-9.6$ |
| 100 | 2 | $-0.0$ | 0.91 | 1.1 | $-0.1$ | 0.91 | 1.1 | $-6.0$ |
| 100 | 3 | $+0.8$ | 0.95 | 1.0 | $-0.2$ | 0.93 | 1.1 | $-1.7$ |
| 1,000 | 1.3 | $-0.1$ | 0.84 | 1.5 | $-0.2$ | 0.95 | 1.0 | $-9.6$ |
| 1,000 | 2 | $-0.2$ | 0.94 | 1.0 | $+0.0$ | 0.96 | 1.0 | $-9.3$ |
| 1,000 | 3 | $+0.2$ | 0.95 | 1.0 | $+0.0$ | 0.97 | 1.0 | $-4.4$ |
| 10,000 | 1.3 | $+0.2$ | 0.78 | 1.2 | $+0.0$ | 0.98 | 0.9 | $-11.6$ |
| 10,000 | 2 | $-0.1$ | 0.94 | 1.5 | $+0.0$ | 0.93 | 1.1 | $-10.0$ |
| 10,000 | 3 | $-0.0$ | 0.94 | 1.1 | $+0.0$ | 0.98 | 0.9 | $-5.7$ |

*Table file: `supp_tables/s7_grid.tex`.*

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

## Data drawn from the working model

To examine the independence assumption of the likelihood, with the
priors, sampler and implementation unchanged, the same nine cells were
also fitted to outbreaks drawn from the DDSA working model itself, in
which cases are independent given the parameters:
$K \sim \mathrm{Binomial}(N, \tau)$, infection days drawn independently
from the map’s incidence, and recovery intervals drawn independently
from the DPH (`sim_working_model` in `sim_core.jl`, checked in
`out/sim_check.txt`; `final_step3_working_model.jl`, 100 replicates per
cell). The output shows each cell next to the chain-binomial results,
with Wilson 95% intervals for the coverage:

``` julia
foreach(println, filter(l -> startswith(l, "|") || startswith(l, "All"), readlines(joinpath(D7, "baseline_grid", "out", "final_step3_summary.md"))))
```

| n | True R₀ | Data | Fits | R₀ coverage \[Wilson 95%\] | R₀ k | R₀ bias (%) | Mean IP coverage | Mean IP k | Mean IP bias (%) | R̂ \> 1.01 | div \> 0 |
|----|----|----|----|----|----|----|----|----|----|----|----|
| 100 | 1.3 | working model | 100 | 0.94 \[0.88, 0.97\] | 1.07 | +0.3 | 0.94 \[0.88, 0.97\] | 1.06 | -0.2 | 0 | 0 |
| 100 | 1.3 | chain binomial | 100 | 0.38 \[0.29, 0.48\] | 2.96 | +1.4 | 0.96 \[0.90, 0.98\] | 1.06 | -0.2 | 0 | 0 |
| 100 | 2.0 | working model | 100 | 0.94 \[0.88, 0.97\] | 1.06 | -0.1 | 0.91 \[0.84, 0.95\] | 1.10 | -0.1 | 0 | 2 |
| 100 | 2.0 | chain binomial | 100 | 0.59 \[0.49, 0.68\] | 2.38 | +1.2 | 0.91 \[0.84, 0.95\] | 1.14 | -0.0 | 0 | 2 |
| 100 | 3.0 | working model | 100 | 0.97 \[0.92, 0.99\] | 0.88 | +1.6 | 0.93 \[0.86, 0.97\] | 1.05 | -0.2 | 0 | 0 |
| 100 | 3.0 | chain binomial | 100 | 0.80 \[0.71, 0.87\] | 1.59 | +0.9 | 0.95 \[0.89, 0.98\] | 0.96 | +0.8 | 0 | 0 |
| 1000 | 1.3 | working model | 100 | 0.90 \[0.83, 0.94\] | 1.03 | +0.1 | 0.95 \[0.89, 0.98\] | 1.01 | -0.2 | 0 | 2 |
| 1000 | 1.3 | chain binomial | 100 | 0.28 \[0.20, 0.37\] | 6.04 | +1.1 | 0.84 \[0.76, 0.90\] | 1.45 | -0.1 | 0 | 6 |
| 1000 | 2.0 | working model | 100 | 0.94 \[0.88, 0.97\] | 0.99 | -0.0 | 0.96 \[0.90, 0.98\] | 0.98 | +0.0 | 0 | 7 |
| 1000 | 2.0 | chain binomial | 100 | 0.51 \[0.41, 0.61\] | 2.71 | -0.5 | 0.94 \[0.88, 0.97\] | 1.00 | -0.2 | 0 | 3 |
| 1000 | 3.0 | working model | 100 | 0.97 \[0.92, 0.99\] | 0.96 | +0.5 | 0.97 \[0.92, 0.99\] | 0.96 | +0.0 | 0 | 2 |
| 1000 | 3.0 | chain binomial | 100 | 0.67 \[0.57, 0.75\] | 1.96 | -0.1 | 0.95 \[0.89, 0.98\] | 1.03 | +0.2 | 0 | 3 |
| 10000 | 1.3 | working model | 100 | 0.95 \[0.89, 0.98\] | 1.04 | +0.0 | 0.98 \[0.93, 0.99\] | 0.91 | +0.0 | 1 | 6 |
| 10000 | 1.3 | chain binomial | 100 | 0.17 \[0.11, 0.26\] | 6.93 | -0.2 | 0.78 \[0.69, 0.85\] | 1.20 | +0.2 | 5 | 8 |
| 10000 | 2.0 | working model | 100 | 0.96 \[0.90, 0.98\] | 0.97 | -0.0 | 0.93 \[0.86, 0.97\] | 1.07 | +0.0 | 0 | 4 |
| 10000 | 2.0 | chain binomial | 100 | 0.52 \[0.42, 0.62\] | 2.88 | +0.4 | 0.94 \[0.88, 0.97\] | 1.49 | -0.1 | 3 | 8 |
| 10000 | 3.0 | working model | 100 | 0.95 \[0.89, 0.98\] | 1.03 | -0.0 | 0.98 \[0.93, 0.99\] | 0.90 | +0.0 | 0 | 7 |
| 10000 | 3.0 | chain binomial | 100 | 0.68 \[0.58, 0.76\] | 1.87 | -0.2 | 0.94 \[0.88, 0.97\] | 1.06 | -0.0 | 1 | 9 |

All working-model fits: R₀ coverage 0.947 \[0.930, 0.960\] (852 of 900)

## Baseline grid: all cells

All cells of the baseline grid, including the geometric simulated
infectious period ($r = 1$), spread ratios and convergence, are in
`S7_stress_tests/baseline_grid/out/summary_cb.md`.

## Table S13: the daily step

The effect of the step of the map at $n = 1{,}000$ with the end cut-off
(the continuous-time stochastic SIR outbreaks of Table S10). DDSA was
fitted with records and map at step $\Delta t$. In the supplement table,
each entry is the difference between the posterior medians of DDSA and
of our continuous-time DSA on exact times for the same outbreak, as a
percentage of the true value (mean over the 50 replicates, with its 95%
CI), followed by the coverage at the daily and the finest step and for
DSA (exact):

``` julia
tex_table("s3_dt.tex", DTPHEAD)
```

| True R0 | Δt = 1 d | Δt = 0.5 d | Δt = 0.25 d | Δt = 0.1 d | Coverage, DDSA Δt = 1 d | Coverage, DDSA Δt = 0.1 d | Coverage, DSA (exact) |
|----|----|----|----|----|----|----|----|
| **(a) $R_0$** |  |  |  |  |  |  |  |
| 1.5 | $+0.1$ $[-0.1, 0.4]$ | $+0.1$ $[-0.1, 0.4]$ | $+0.1$ $[-0.1, 0.4]$ | $+0.2$ $[-0.1, 0.4]$ | 0.36 | 0.36 | 0.42 |
| 2 | $+0.7$ $[0.4, 1.0]$ | $+0.3$ $[0.0, 0.6]$ | $+0.1$ $[-0.2, 0.4]$ | $+0.1$ $[-0.2, 0.3]$ | 0.42 | 0.46 | 0.52 |
| 4 | $+7.2$ $[6.6, 7.9]$ | $+3.4$ $[2.8, 4.0]$ | $+1.7$ $[1.2, 2.3]$ | $+0.8$ $[0.2, 1.3]$ | 0.54 | 0.80 | 0.84 |
| **(b) Mean infectious period** |  |  |  |  |  |  |  |
| 1.5 | $+2.8$ $[2.4, 3.1]$ | $+1.1$ $[0.9, 1.3]$ | $+0.5$ $[0.3, 0.7]$ | $+0.2$ $[0.0, 0.4]$ | 0.84 | 0.86 | 0.82 |
| 2 | $+2.1$ $[1.8, 2.4]$ | $+0.6$ $[0.3, 0.8]$ | $+0.1$ $[-0.1, 0.3]$ | $-0.1$ $[-0.3, 0.1]$ | 0.74 | 0.84 | 0.84 |
| 4 | $+1.4$ $[1.1, 1.7]$ | $+0.2$ $[-0.1, 0.4]$ | $-0.1$ $[-0.3, 0.1]$ | $-0.1$ $[-0.4, 0.1]$ | 0.90 | 0.98 | 0.96 |

*Table file: `supp_tables/s3_dt.tex`.*

Bias of the posterior median and coverage of $R_0$ at every step, with
DSA (exact) and the complete-data benchmark (the same table for the mean
infectious period is under *Mean infectious period* in S3):

``` julia
tex_table("s3_dt_full.tex", DTHEAD)
```

| True R0 | DDSA, Δt = 1 d | Δt = 0.5 d | Δt = 0.25 d | Δt = 0.1 d | DSA (exact times) | Complete data |
|----|----|----|----|----|----|----|
| **(a) Relative bias of the posterior median (%)** |  |  |  |  |  |  |
| 1.5 | $-0.9$ | $-0.9$ | $-0.9$ | $-0.9$ | $-1.0$ | $-0.9$ |
| 2 | $+0.7$ | $+0.3$ | $+0.1$ | $+0.1$ | $+0.0$ | $+0.6$ |
| 4 | $+6.1$ | $+2.2$ | $+0.6$ | $-0.4$ | $-1.2$ | $-0.9$ |
| **(b) Coverage of the 95% CrI** |  |  |  |  |  |  |
| 1.5 | 0.36 | 0.36 | 0.36 | 0.36 | 0.42 | 0.96 |
| 2 | 0.42 | 0.46 | 0.46 | 0.46 | 0.52 | 0.96 |
| 4 | 0.54 | 0.80 | 0.82 | 0.80 | 0.84 | 0.96 |

*Table file: `supp_tables/s3_dt_full.tex`.*

## Table S14: the GLCT stage cap

We use the synthetic setting of Di Lauro et al. (a Gamma infectious
period with mean 9 d and shape 13.5, and $N = 10{,}000$), with outbreaks
simulated by the exact event-driven method of Table S10 and recorded by
day, to see the effect of the minimum duration of the NegBin DPH law.
The data are the full line list (A), infection times only (B) or
recovery intervals only (C). DDSA was fitted with the GLCT structure and
the NegBin DPH law, and with the Boxcar structure and the discretised
Gamma law, which is the generating family. We compare both with our
implementation of continuous-time DSA with a Gamma infectious period, on
exact times.

The scripts are in `S3_S6_dsa_comparison/`: `s5_nonexp.jl` runs the main
fits; `s5_0_checks.jl` checks the convergence of the renewal grid,
compares the discretised Gamma with the simulated intervals and looks at
the shortest intervals (log in `out/logs/s5_0_checks.log`); and
`s5_boxcar_dt.jl` refits the Boxcar arm at $\Delta t = 0.5$ and 0.25 d,
and is run separately.

``` julia
./run_method_supplement.sh S5
DTS=0.5,0.25 REPS=20 SCEN=A BUDGET=30000 RUN=s5_boxcar_dt julia --project=@. -t 10 S3_S6_dsa_comparison/s5_boxcar_dt.jl
```

Each table summarises 20 replicates per row. The true values are
$R_0 = 2.25$, a mean infectious period of 9 d and an SD of 2.45 d. $R_0$
and the mean infectious period each have three columns (bias, coverage
and $k$), and the IP SD the first two:

- **Bias:** the difference between the posterior median and the true
  value, relative to the true value, in %, averaged over the replicates.
- **Coverage:** the proportion of replicates whose 95% CrI contains the
  true value.
- **Spread ratio $k$:** the SD of the posterior medians over replicates
  divided by the mean posterior SD. It is close to 1 when the intervals
  match the variability between replicates, and above 1 when they are
  too narrow.

For example, the first row’s $R_0$ columns, +10.4, 0.00 and 5.5, mean
that the posterior median of $R_0$ was on average 10.4% above 2.25, that
none of the 20 intervals contained 2.25, and that the medians varied 5.5
times more between replicates than the posterior SDs indicate. “Gamma
shape” is the posterior median of the shape of the Gamma infectious
period (Boxcar and continuous-time DSA). For the GLCT (NegBin DPH)
model, $r$ is the posterior median number of stages and $r_{\max}$ the
largest $r$ allowed, the shortest recorded interval; $r$ and the Gamma
shape are different parameters. The Gamma shape, $r$ and $r_{\max}$ are
medians over replicates. The GLCT fits of scenario B did not converge
(marked †). Table S14:

``` julia
conv(CSV.read(joinpath(D3, "s5_main.csv"), DataFrame))
println()
const S5HEAD = ["Scenario", "Model", "R0: bias %", "R0: cov.", "R0: k", "Mean IP: bias %", "Mean IP: cov.", "Mean IP: k",
                "IP SD: bias %", "IP SD: cov.", "Gamma shape", "r (r_max)"]
tex_table("s5_nonexp_supp.tex", S5HEAD)
```

*Convergence: 180 fits; 30 with R̂ \> 1.01, 22 with at least one
divergent transition (all kept in the summaries).*

| Scenario | Model | R0: bias % | R0: cov. | R0: k | Mean IP: bias % | Mean IP: cov. | Mean IP: k | IP SD: bias % | IP SD: cov. | Gamma shape | r (r_max) |
|----|----|----|----|----|----|----|----|----|----|----|----|
| A | DDSA, GLCT (NegBin DPH) | $+10.4$ | 0.00 | 5.5 | $-3.0$ | 0.05 | 3.9 | $+104.3$ | 0.00 | – | 2 (2) |
| A | DDSA, Boxcar (Gamma) | $+3.8$ | 0.00 | 2.7 | $-0.3$ | 0.85 | 1.1 | $-0.2$ | 0.90 | 13.5 | – |
| A | DSA (Gamma) | $-0.3$ | 0.40 | 2.7 | $+0.0$ | 0.95 | 1.1 | $+0.3$ | 0.90 | 13.5 | – |
| B | DDSA, GLCT (NegBin DPH) |  |  |  |  |  |  |  |  |  |  |
| B | DDSA, Boxcar (Gamma) | $+0.2$ | 0.85 | 1.3 | $-16.5$ | 0.05 | 1.8 | $+19.8$ | 0.70 | 6.5 | – |
| B | DSA (Gamma) | $+0.3$ | 0.85 | 1.3 | $-0.1$ | 0.75 | 1.6 | $+5.5$ | 0.85 | 14.8 | – |
| C | DDSA, GLCT (NegBin DPH) | – | – | – | $-0.0$ | 1.00 | 0.5 | $+112.6$ | 0.00 | – | 2 (2) |
| C | DDSA, Boxcar (Gamma) | – | – | – | $-0.0$ | 0.90 | 1.0 | $+0.2$ | 0.90 | 13.5 | – |
| C | DSA (Gamma) | – | – | – | $-0.0$ | 0.95 | 1.0 | $+0.2$ | 0.95 | 13.4 | – |

*Table file: `supp_tables/s5_nonexp_supp.tex`.*

## The NegBin DPH law without the cap

An $r$-stage NegBin DPH, built stage by stage with at most one
transition per day, gives zero probability to intervals shorter than $r$
days. For each $r$, the table gives the NegBin DPH that best fits the
recorded Gamma interval law of Table S14 (mean 9 d, shape 13.5, recorded
by day) in the large-sample limit, ignoring the intervals shorter than
$r$, and the share of recorded intervals it excludes. No data or fits
are used. Script: `S3_S6_dsa_comparison/s5_negbin_uncapped.py` (runs in
the S5 step).

``` julia
print(read(joinpath(D3, "s5_negbin_uncapped.md"), String))
```

Recorded Gamma law: mean 9.00 d, SD 2.48 d (SD of the continuous Gamma
2.45 d); share of recorded intervals of 3 d or less 2.3e-03.

| Stages $r$ | Share of recorded intervals shorter than $r$ | Mean (d) | SD (d) | SD bias (%) |
|----|----|----|----|----|
| 1 | 2.3e-12 | 9.00 | 8.49 | +246.4 |
| 2 | 1.3e-06 | 9.00 | 5.61 | +129.1 |
| 3 | 1.3e-04 | 9.00 | 4.24 | +73.2 |
| 4 | 2.3e-03 | 9.01 | 3.36 | +37.2 |
| 5 | 1.6e-02 | 9.08 | 2.72 | +11.2 |
| 6 | 5.9e-02 | 9.27 | 2.25 | -8.2 |
| 7 | 1.5e-01 | 9.62 | 1.90 | -22.4 |
| 8 | 2.9e-01 | 10.14 | 1.65 | -32.8 |
| 9 | 4.5e-01 | 10.78 | 1.46 | -40.5 |
| 10 | 6.1e-01 | 11.51 | 1.32 | -46.1 |

With five or six stages the NegBin DPH is close to the recorded Gamma
law, but it excludes about 2 and 6% of the recorded intervals. In the
fits of Table S14 the shortest recorded interval was 2 d in 11 of the 20
replicates, 3 d in 8 and 1 d in one, so $r$ was capped at 1–3.

## The remaining Boxcar $R_0$ bias and the time step

The Boxcar structure recovers the shape of the infectious period. Its
remaining $R_0$ bias halves each time we halve the step, consistent with
an effect of the map (M):

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

# S5 Reproducibility

All runs use the shared environment at the root of the repository (Julia
1.11, Turing.jl 0.49; `Manifest.toml`), which `--project=@.` finds from
this folder, and the repository’s phase-type and integer-parameter
modules, which the scripts find by searching the parent folders. A
single script reruns every section, with a time cap on each process, and
writes the tables of the supplement to `supp_tables/`:

``` julia
# from this folder
./run_method_supplement.sh                # S2 S3 S5 VALID S7 TABLES FIGURES (script section names)
./run_method_supplement.sh S2 S7          # selected sections
DRY=1 ./run_method_supplement.sh          # print the commands only
python3 -I make_supplement_tables.py      # tables only: supp_tables/*.tex
```

## Numerical validation and software

The numerical checks and software versions behind supplement S5.3.

**Likelihood forms.** The DDSA fits of supplement Sections S3 and S4
evaluate the main-text likelihood on the log scale
($\log S_k = \log S_0 - \sum_{j<k} \beta I_j$, new infections from
$\log(-\mathrm{expm1}(-\beta I_k))$). This form needs no numerical
guards in the far tails that these thousands of fits visit during
warm-up, and near every main-text posterior it agrees with the main-text
form to within $6\times10^{-9}$. The Poisson factorisation of supplement
Eq. (1) holds numerically to $10^{-13}$, and the count model with a
never-infected cell equals infection-only DDSA (Eq. (2)) to
$1.2\times10^{-13}$ in log-likelihood.

**Grid posteriors.** The grids of S2 use unequal steps in the mean
infectious period, finer below 15 d. Their log densities equal the
Turing log-joints of the same models to within $3\times10^{-11}$ for the
discrete map and $6\times10^{-7}$ for the ODE (its solver tolerance).
Halving the grid steps changed the posterior medians and 95% limits of
$R_0$ by at most 0.05 in the attack-rate sweep at $N = 99$. It left the
map–Poisson posterior at $N = 10{,}000$ unchanged to two decimals and,
over the range of the mean infectious period that holds the ODE–Poisson
posterior at $N = 10{,}000$ (6–13 d), changed its medians and 95% limits
by at most 0.05. Extending the ODE’s support from 20 to 40 stages
changed its posterior medians by at most 0.2 for $R_0$ and 0.5 d for the
mean infectious period. The individual checks are in S2 (*Numerical grid
posteriors: checks*, and the checks below Table S8).

**Continuous-time solvers.** The continuous-time mean field of S3 was
integrated in log coordinates by a fourth-order Runge–Kutta scheme with
steps of at most 0.02 d, stepping exactly to every infection time. It
agrees with a ninth-order adaptive solver (OrdinaryDiffEq.jl, Vern9) to
within $10^{-9}$ in the log-likelihood
(`S3_S6_dsa_comparison/out/logs/s3_0_checks.log`), and with the Euler
scheme of the published SDS code (steps of 0.01 d) to within a few log
units over the parameter points checked (S3, *Table S9: published codes
in their own settings*). As $\Delta t \to 0$, DDSA log-likelihood ratios
approach the continuous-time ones: for one outbreak with $n = 1{,}000$,
2.89, 5.07, 6.57, 7.48 and 7.98 at $\Delta t = 1$, 0.5, 0.25, 0.1 and
0.02 d, against 8.12 for the continuous-time value with DDSA’s
final-size convention (8.23 with the per-susceptible convention of
KhudaBukhsh et al.). The Gamma renewal scheme, on a 0.1 d grid, gives
log-likelihood ratios within 0.02 of a 0.025 d grid
(`out/logs/s5_0_checks.log`), and with the initial condition of the
published PDE code its likelihood of infection times agrees with that
code on the 10,000-point grid of their study (Table S9).

**Software.** Our Julia code used Julia 1.11.6 with Turing.jl 0.49.0
(NUTS), ForwardDiff.jl 1.4.6 (gradients), Distributions.jl 0.25.131,
SpecialFunctions.jl and MCMCDiagnosticTools.jl 0.3.19 ($\hat R$, ESS),
with `IntegerParameters.jl` for summed-out integer parameters. All
versions are in the repository’s root `Manifest.toml`; OrdinaryDiffEq.jl
7.8.1 is used only to check the Runge–Kutta solver. The published code
was run unchanged. SDS.Epidemic (R package SDSMCMC 0.1.0;
<https://github.com/cbskust/SDS.Epidemic>, commit `3ecb29d`) was run
with R 4.6.1, plyr 1.8.9, MASS 7.3-65, smfsb 1.5, abind 1.4-8, ramcmc
0.1.2 and Rcpp 1.1.2 (`sds_r_versions.txt`), using `SDS.MCMC` with the
priors of its example scripts and four chains of 10,000 iterations
(2,000 burn-in) for Table S9. DSA_refactor
(<https://github.com/Zkeggia/DSA_refactor>, commit `26671d6`) was run
with Python 3.14.6, numpy 2.5.3, scipy 1.18.1, pyswarms 1.3.0 and pydoe
1.5.0 (`dsa_refactor_requirements.txt`; imported under its former name
pyDOE through a one-line alias outside their code), with the generator,
data extraction, bounds and particle-swarm optimiser (100 particles ×
100 iterations) of its synthetic study for Table S9.
