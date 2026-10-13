# Supplement S5 checks (numerical claims in Section S5 and Table S9):
#  1. the continuous-time Gamma DSA likelihood (renewal scheme, grid h = 0.1 d) against finer grids (h = 0.05, 0.025 d):
#     log-likelihood ratios between parameter points, scenarios A and B (S5 states "within 0.02 of a 0.025 d grid");
#  2. the discretised-Gamma interval law of the Boxcar arm against the recorded intervals of simulated outbreaks;
#  3. the shortest recorded interval (the GLCT stage cap) and the number of short intervals in each replicate.
# Output: out/logs/s5_0_checks.log
# Usage (from the supplementary folder): julia --project=@. S3_S6_dsa_comparison/s5_0_checks.jl
include(joinpath(@__DIR__, "s5_nonexp.jl"))
using Printf, Dates
const LOGF = open(joinpath(OUTDIR, "logs", "s5_0_checks.log"), "w")
say(x) = (println(x); println(LOGF, x); flush(LOGF))
say("S5 checks ($(Dates.now()))")

# 1. grid convergence of the continuous-time Gamma DSA likelihood (replicate 1)
sim = simulate5(1); T = ceil(maximum(vcat(sim.rec_t, sim.seed_rec))) + 1.0
de = obs_exact_w(sim, T)
p1 = (0.25, 9.0, 13.5, 0.005); p2 = (0.26, 8.7, 12.0, 0.006); p3 = (0.24, 9.3, 15.0, 0.004)
say("\n1. Continuous-time Gamma DSA: log-likelihood ratios against the true parameters (β, μ, k, ρ) = $p1")
worst = 0.0
for use in (:A, :B)
    lr = Dict{Float64, Tuple{Float64, Float64}}()
    for h in (0.1, 0.05, 0.025)
        l1 = dsa_gamma_loglik(p1..., de, use; h = h)
        lr[h] = (dsa_gamma_loglik(p2..., de, use; h = h) - l1, dsa_gamma_loglik(p3..., de, use; h = h) - l1)
        say(@sprintf("   scenario %s, h = %.3f d: LR(p2) = %9.4f  LR(p3) = %9.4f", use, h, lr[h]...))
    end
    d = maximum(abs.(collect(lr[0.1]) .- collect(lr[0.025])))
    global worst = max(worst, d)
    say(@sprintf("   scenario %s: max |LR(h = 0.1) − LR(h = 0.025)| = %.4f", use, d))
end
say(@sprintf("   overall: %.4f (S5 states within 0.02)", worst))

# 2. discretised-Gamma law against simulated recorded intervals (cases, replicates 1–5)
cd = Int[]
for rep in 1:5
    s = simulate5(rep)
    append!(cd, [max(1, floor(Int, r) - floor(Int, t)) for (t, r) in zip(s.inf_t, s.rec_t)])
end
pc, _ = discretised_gamma(μ5, K5, 40)
emp = [count(==(d), cd) / length(cd) for d in 1:40]
say("\n2. Discretised Gamma (mean 9 d, shape 13.5) against $(length(cd)) recorded case intervals, replicates 1–5")
say("   δ   simulated   discretised law")
for d in 3:16; say(@sprintf("   %2d   %.5f     %.5f", d, emp[d], pc[d])); end
say(@sprintf("   mean δ: simulated %.3f d, law %.3f d; max |difference| over δ = %.4f", mean(cd), sum((1:40) .* pc), maximum(abs.(emp .- pc))))
say(@sprintf("   mean within-day position of infection times (replicate 1): %.3f (uniform: 0.5)", mean(t - floor(t) for t in sim.inf_t)))

# 3. shortest recorded interval per replicate (the GLCT cap r ≤ min δ) and short intervals
say("\n3. Shortest recorded interval and number of intervals ≤ 3 d, per replicate")
mins = Int[]
for rep in 1:20
    s = simulate5(rep); TT = ceil(maximum(vcat(s.rec_t, s.seed_rec))) + 1.0
    db = obs_steps_box(s, TT); push!(mins, db.du[1])
    say(@sprintf("   rep %2d: K = %d, min δ = %d, n(δ ≤ 3) = %d", rep, db.K, db.du[1], sum(db.dn[db.du .<= 3])))
end
say("   min δ across replicates: " * join(["$m: $(count(==(m), mins))" for m in sort(unique(mins))], ", "))
close(LOGF)
