# Step 3 summary: coverage, spread ratio and bias of the DPH model on data drawn from the working model
# (final_step3_working_model.csv), next to the same cells of the chain-binomial grid (results_cb.csv, r = 4).
# Output: out/final_step3_summary.md
# Usage (from the supplementary folder): julia --project=@. S7_stress_tests/baseline_grid/final_step3_summarise.jl
include(joinpath(@__DIR__, "sim_core.jl"))
using CSV, DataFrames
wm = CSV.read(joinpath(OUTDIR, "final_step3_working_model.csv"), DataFrame)
cb = CSV.read(joinpath(OUTDIR, "results_cb.csv"), DataFrame)
cb = cb[(cb.r .== 4) .& (cb.model .== "dph"), :]
out = open(joinpath(OUTDIR, "final_step3_summary.md"), "w"); say(x) = (println(x); println(out, x))
"Wilson 95% interval for a proportion x/n."
function wilson(x, n; z = 1.959964)
    p = x / n; c = (p + z^2 / (2n)) / (1 + z^2 / n); h = z * sqrt(p * (1 - p) / n + z^2 / (4n^2)) / (1 + z^2 / n)
    (c - h, c + h)
end
function st(s, p, tr)
    t = s[!, tr]; m = s[!, "$(p)_med"]; lo = s[!, "$(p)_lo"]; hi = s[!, "$(p)_hi"]
    x = count((lo .<= t) .& (t .<= hi)); w = wilson(x, nrow(s))
    (cov = x / nrow(s), lo = w[1], hi = w[2], k = std(m) / mean((hi .- lo) ./ 3.92), bias = 100 * mean((m .- t) ./ t))
end
say("## Step 3: data from the working model against chain-binomial data (r = 4, DPH model, data-based starts)\n")
say("Coverage of the 95% CrI with its 95% Wilson interval; k = SD of the posterior medians / mean posterior SD (from the interval width); bias = mean relative bias of the posterior median (%).\n")
say("| n | True R₀ | Data | Fits | R₀ coverage [Wilson 95%] | R₀ k | R₀ bias (%) | Mean IP coverage | Mean IP k | Mean IP bias (%) | R̂ > 1.01 | div > 0 |")
say("|---|---|---|---|---|---|---|---|---|---|---|---|")
for N in (100, 1000, 10_000), R0 in (1.3, 2.0, 3.0)
    for (lab, df) in (("working model", wm), ("chain binomial", cb))
        s = df[(df.N .== N) .& (df.R0 .== R0), :]
        nrow(s) == 0 && continue
        a = st(s, "R0", "R0_true"); b = st(s, "ip", "ip_true")
        say(@sprintf("| %d | %.1f | %s | %d | %.2f [%.2f, %.2f] | %.2f | %+.1f | %.2f [%.2f, %.2f] | %.2f | %+.1f | %d | %d |", N, R0, lab, nrow(s),
                     a.cov, a.lo, a.hi, a.k, a.bias, b.cov, b.lo, b.hi, b.k, b.bias, count(>(1.01), s.rhat), count(>(0), s.div)))
    end
end
x = count((wm.R0_lo .<= wm.R0_true) .& (wm.R0_true .<= wm.R0_hi)); w = wilson(x, nrow(wm))
say(@sprintf("\nAll working-model fits: R₀ coverage %.3f [%.3f, %.3f] (%d of %d)", x / nrow(wm), w[1], w[2], x, nrow(wm)))
close(out)
