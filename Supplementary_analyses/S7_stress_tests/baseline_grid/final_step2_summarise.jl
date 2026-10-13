# Step 2 summary: coverage, spread ratio and bias for R0 and β by variant and R0; sanity check of the final-size variance.
include(joinpath(@__DIR__, "sim_core.jl"))
using CSV, DataFrames
df = CSV.read(joinpath(OUTDIR, "final_step2_term_attribution.csv"), DataFrame)
out = open(joinpath(OUTDIR, "final_step2_summary.md"), "w"); say(x) = (println(x); println(out, x))
say("## Step 2: term attribution (chain binomial, r = 4, N = 1,000, 100 replicates per R₀; DPH model, final code, data-based starts)\n")
say("| Variant | True R₀ | R₀ coverage | R₀ spread ratio | R₀ bias | β coverage | β spread ratio | β bias | R̂ > 1.01 | div > 0 |")
say("|---|---|---|---|---|---|---|---|---|---|")
lab = Dict("binom" => "(a) Binomial final size", "none" => "(b) no final-size term", "normal" => "(c) Normal, corrected variance")
for v in ("binom", "none", "normal"), R0 in (1.3, 2.0, 3.0)
    s = df[(df.variant .== v) .& (df.R0 .== R0), :]
    st(p, tr) = (t = s[!, tr]; m = s[!, "$(p)_med"]; lo = s[!, "$(p)_lo"]; hi = s[!, "$(p)_hi"];
                 (mean((lo .<= t) .& (t .<= hi)), std(m) / mean((hi .- lo) ./ 3.92), mean((m .- t) ./ t)))
    a = st("R0", "R0_true"); b = st("beta", "beta_true")
    say(@sprintf("| %s | %.1f | %.2f | %.2f | %+.3f | %.2f | %.2f | %+.3f | %d | %d |", lab[v], R0, a..., b..., count(>(1.01), s.rhat), count(>(0), s.div)))
end
say("\n### Sanity check of the final-size variance (not used for the decision)\n")
say("| True R₀ | Empirical Var(K/N) over 100 replicates | Binomial τ(1−τ)/N | Corrected σ²_Z/N | ratio empirical / corrected |")
say("|---|---|---|---|---|")
for R0 in (1.3, 2.0, 3.0)
    Ks = Float64[]
    for rep in 1:100
        base = hash(("cb", 1000, R0, 4, rep)); a = 0
        while true; a += 1; s = sim_chainbinomial(Xoshiro(base + UInt(a)), 1000, R0, 4); s.K >= 100 && (push!(Ks, s.K / 1000); break); end
    end
    dph = PhaseTypeDistributions.dph_negative_binomial(4, 4 / WBAR); _, lt, _ = glct_log(dph.S, dph.α, R0 / WBAR, 1 / 1001, 3000)
    τ = exp(lt); vb = τ * (1 - τ) / 1000; vc = fs_var(τ, R0, (1 - 4 / WBAR) / 4) / 1000
    say(@sprintf("| %.1f | %.2e | %.2e | %.2e | %.2f |", R0, var(Ks), vb, vc, var(Ks) / vc))
end
close(out)
