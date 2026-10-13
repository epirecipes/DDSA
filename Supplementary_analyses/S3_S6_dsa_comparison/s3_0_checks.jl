# Checks for s3_core.jl (log only): simulator against the mean field, RK4 DSA likelihood against OrdinaryDiffEq, DDSA
# likelihood ratios approaching the continuous-time DSA ones as Δt → 0, and one timed fit of each model.
# Usage (from the supplementary folder): julia --project=@. -t 4 S3_S6_dsa_comparison/s3_0_checks.jl
include(joinpath(@__DIR__, "s3_core.jl"))
using OrdinaryDiffEq
say(x...) = (println(x...); flush(stdout))

# 1. simulator: mean attack rate among susceptibles against the mean-field final size
for R0 in (1.5, 2.0, 4.0)
    n = 10_000; m = round(Int, 0.05n); μ = 5.0
    ks = [length(sim_outbreak(Xoshiro(i), n, m, R0 / μ, Exponential(μ)).inf_t) / n for i in 1:40]
    mf = meanfield_cutoffs(R0, μ, m / (n + m))
    say(@sprintf("sim R0=%.1f: mean attack rate %.4f (sd %.4f) vs mean field %.4f; t_peak %.1f d, t99 %.1f d",
                 R0, mean(ks), std(ks), mf.final, mf.tpeak, mf.t99))
end

# 2. RK4 log-coordinates likelihood against Vern9 at tight tolerance
n = 1000; m = 50; μ = 5.0; R0 = 2.0; β = R0 / μ
sim = sim_outbreak(Xoshiro(7), n, m, β, Exponential(μ))
mf = meanfield_cutoffs(R0, μ, m / (n + m)); Tend = ceil(mf.t99); Tpk = ceil(mf.tpeak)
de = obs_exact(sim, Tend)
function dsa_ref(β, μ, ρ, d)
    f!(du, u, p, t) = (du[1] = -β * u[1] * u[2]; du[2] = β * u[1] * u[2] - u[2] / μ)
    sol = solve(ODEProblem(f!, [1 - ρ, ρ], (0.0, d.T)), Vern9(); reltol = 1e-13, abstol = 1e-15, saveat = vcat(d.ti, d.T))
    S = sol[1, :]; I = sol[2, :]; S0 = 1 - ρ
    ll = sum(log(β) + log(S[i]) + log(I[i]) - log(S0) for i in 1:d.K) + (d.n - d.K) * (log(S[end]) - log(S0))
    ll + d.nrec * log(1 / μ) - d.expo / μ
end
for (b, mu, r) in ((β, μ, 0.0476), (0.35, 6.0, 0.03), (0.5, 4.0, 0.08))
    a = dsa_loglik(b, mu, r, de); a2 = dsa_loglik(b, mu, r, de; h = 0.005); c = dsa_ref(b, mu, r, de)
    say(@sprintf("DSA loglik at (β=%.2f, μ=%.1f, ρ=%.3f): RK4 h=0.02 %.8f, h=0.005 %.8f, Vern9 %.8f (diff %.1e)", b, mu, r, a, a2, c, a - c))
end

# 3. log-likelihood ratios between two parameter points: DDSA at Δt → 0 against continuous-time DSA (exact times)
θ1 = (β, μ, 0.0476); θ2 = (0.37, 5.5, 0.045)
cr = dsa_loglik(θ2..., de) - dsa_loglik(θ1..., de)
for Δt in (1.0, 0.5, 0.25, 0.1, 0.02)
    ds = obs_steps(sim, Tend; Δt)
    say(@sprintf("Δt=%.2f: DDSA log LR %.4f vs DSA %.4f", Δt, ddsa_loglik(θ2..., ds) - ddsa_loglik(θ1..., ds), cr))
end

# 4. gradient works, and one timed fit of each model (end cut-off)
using ForwardDiff
dd = obs_steps(sim, Tend); dj = obs_jitter(Xoshiro(1), sim, Tend)
say("gradients: ", ForwardDiff.gradient(x -> ddsa_loglik(x..., dd), [β, μ, 0.0476]), "  ", ForwardDiff.gradient(x -> dsa_loglik(x..., de), [β, μ, 0.0476]))
for (nm, mdl, d) in (("ddsa", ddsa_model(dd), dd), ("dsa exact", dsa_model(de), de), ("dsa jitter", dsa_model(dj), dj))
    r = fit_bayes(Xoshiro(3), mdl, d, dd.meanδ, :end)
    say(@sprintf("%-10s %5.1f s  β %.3f [%.3f, %.3f]  μ %.2f [%.2f, %.2f]  R0 %.3f [%.3f, %.3f]  ρ %.4f  R̂ %.4f ESS %.0f/%.0f div %d",
                 nm, r.time, r.β[3], r.β[4], r.β[5], r.μ[3], r.μ[4], r.μ[5], r.R0[3], r.R0[4], r.R0[5], r.ρ[3], r.rhat, r.ess, r.ess_tail, r.div))
end
r = fit_exact(Xoshiro(3), sim, Tend)
say(@sprintf("exact      β %.3f [%.3f, %.3f]  μ %.2f [%.2f, %.2f]  R0 %.3f [%.3f, %.3f]", r.β[3], r.β[4], r.β[5], r.μ[3], r.μ[4], r.μ[5], r.R0[3], r.R0[4], r.R0[5]))
say("truth: β $β, μ $μ, R0 $R0, ρ $(m/(n+m)); K = $(de.K), T = $Tend, peak cut-off $Tpk")
