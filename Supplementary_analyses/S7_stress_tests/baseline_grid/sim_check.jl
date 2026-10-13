# Checks for the S7 baseline grid:
#  (1) the type-stable map and grouped likelihood equal the notebook's simulate_discrete_glct / dph_full_loglik;
#  (2) the chain-binomial simulator's mean final size and incidence match the deterministic map at large N;
#  (3) Gillespie final sizes against the continuous final-size relation;
#  (4) timing of single fits (to plan the full study).
include(joinpath(@__DIR__, "sim_core.jl"))
const OUT = open(joinpath(OUTDIR, "sim_check.txt"), "w")
say(x) = (println(x); println(OUT, x); flush(OUT))

# ---- (1) notebook functions, verbatim
function rate_to_proportion(r, t=1.0)
    if r < 0 || isnan(r); return 0.0 elseif isinf(r); return 1.0
    else result = 1 - exp(-r * t); return isnan(result) ? 0.0 : max(min(result, 1.0), 0.0) end
end
function simulate_discrete_glct(Tm::AbstractMatrix, α::AbstractVector, β::Real, ρ::Real, nsteps::Int, Δt::Real)
    k = length(α); S = 1.0 - ρ; x = α .* ρ; R = zero(ρ)
    S_hist = Any[S]; I_hist = Any[sum(x)]; R_hist = Any[R]
    for _ in 1:nsteps
        I = sum(x); p_inf = rate_to_proportion(β * I, Δt); new_inf = p_inf * S
        x = Tm' * x + α .* new_inf
        absorption_probs = ones(eltype(α), k) - Tm * ones(eltype(α), k)
        recov = dot(absorption_probs, x); S = S - new_inf; R = R + recov
        push!(S_hist, S); push!(I_hist, sum(x)); push!(R_hist, R)
    end
    τ = S_hist[1] - S_hist[end]; dS = diff(S_hist); f = max.(-dS, 0.0); sF = sum(f)
    f = sF > 0 ? f ./ sF : fill(one(eltype(f))/length(f), length(f))
    return (τ = τ, f = f)
end
function dph_full_loglik_nb(nb_r, β, ρ, nb_p, N, M, nsteps, K, th, delta, Δt)
    any(<(nb_r), delta) && return -Inf
    dph = PhaseTypeDistributions.dph_negative_binomial(nb_r, nb_p)
    result = simulate_discrete_glct(dph.S, dph.α, β, ρ, nsteps, Δt)
    τ_valid = clamp(result.τ, 0.001, 0.999); f_safe = safe_normalise(result.f, nsteps)
    ll = logpdf(Binomial(N, τ_valid), K)
    if K > 0
        all(x -> isfinite(x) && !isnan(x), f_safe) && abs(sum(f_safe) - 1) < 0.01 || return -Inf
        ll += logpdf(Multinomial(K, f_safe), th)
    end
    return ll + sum(logpdf(dph, delta[i]) for i in 1:(M + K))
end
# (1a) the guarded copy equals the notebook function at arbitrary points, including ones where the guards bind;
# (1b) the log-scale likelihood used in the fits equals the guarded one near the generating values (guards inactive)
rng = Xoshiro(1)
md = 0.0; ml = 0.0; mt = 0.0
for (N, R0, r) in ((100, 2.0, 4), (1000, 1.3, 1), (10000, 3.0, 4))
    s = sim_chainbinomial(rng, N, R0, r); d = linelist(s, N)
    for (rr, β, ρ, p) in ((r, 0.3, 0.001, 0.5), (1, 0.2, 0.01, 0.2), (min(3, d.du[1]), 0.5, 1e-4, 0.7))
        a = full_loglik_guarded(rr, β, ρ, p, d); b = dph_full_loglik_nb(rr, β, ρ, p, N, M_SEED, d.nsteps, d.K, d.th, d.delta, 1.0)
        global md = max(md, abs(a - b)); global mt = max(mt, abs(full_loglik(rr, β, ρ, p, d) - b))
    end
    for _ in 1:50
        z() = exp(0.05 * randn(rng)); β = z() * R0 / WBAR; p = min(0.99, z() * r / WBAR); ρ = z() / (N + 1)
        global ml = max(ml, abs(full_loglik(r, β, ρ, p, d) - full_loglik_guarded(r, β, ρ, p, d)))
    end
end
say("(1a) max |guarded copy − notebook dph_full_loglik| over 9 test points: $md")
say("(1b) max |log-scale − guarded| over 150 points within ±5% of the generating values: $ml")
say("     (for comparison, at the 9 test points of (1a), where the guards bind at some: $mt)")

# ---- (2) simulator mean vs map (N = 10,000)
for (R0, r) in ((2.0, 4), (2.0, 1), (1.3, 4))
    N = 10_000; sims = [sim_chainbinomial(Xoshiro(100 + i), N, R0, r) for i in 1:400]
    maj = filter(s -> s.K >= 0.1N, sims)
    dph = PhaseTypeDistributions.dph_negative_binomial(r, r / WBAR)
    τ, _ = glct_tau_f(dph.S, dph.α, R0 / WBAR, M_SEED / (N + M_SEED), 3000)
    say("(2) N=10,000 R0=$R0 r=$r: major-outbreak share $(round(length(maj)/400, digits=3)) (branching-process prediction ≈ $(round(1 - 1/R0, digits=3)) for r=1 seeds); " *
        "mean K/N in major outbreaks $(round(mean(s.K for s in maj)/N, digits=4)) vs map τ·(N+M)/N $(round(τ*(N+M_SEED)/N, digits=4)); mean recorded δ $(round(mean(mean(linelist(s, N).delta) for s in maj[1:20]), digits=3)) vs W̄ $(round(WBAR, digits=3))")
end
# ---- (3) Gillespie vs continuous final-size relation z = 1 − exp(−R0 z)
for (R0, r) in ((2.0, 4), (1.3, 1))
    N = 10_000; sims = [sim_gillespie(Xoshiro(200 + i), N, R0, r) for i in 1:60]
    maj = filter(s -> s.K >= 0.1N, sims)
    z = 0.5; for _ in 1:200; z = 1 - exp(-R0 * z); end
    say("(3) Gillespie N=10,000 R0=$R0 r=$r: major share $(round(length(maj)/60, digits=2)); mean K/N $(round(mean(s.K for s in maj)/N, digits=4)) vs continuous final size $(round(z, digits=4)); mean δ $(round(mean(mean(linelist(s, N).delta) for s in maj[1:min(10,end)]), digits=3))")
end
# ---- (4) timing of single fits (4 chains × 1000 + 1000 warm-up, serial chains, one fit per thread)
get(ENV, "TIMING", "0") == "1" || (close(OUT); exit())
jobs = [(N, R0, r, mdl) for N in (100, 1000, 10000) for (R0, r) in ((2.0, 4), (1.3, 4)) for mdl in (get(ENV, "TIMING_MODELS", "dph,geo") |> x -> Symbol.(split(x, ",")))]
lk = ReentrantLock()
Threads.@threads for j in eachindex(jobs)
    N, R0, r, mdl = jobs[j]
    s = sim_chainbinomial(Xoshiro(7), N, R0, r); k = 0
    while s.K < 0.1N; k += 1; s = sim_chainbinomial(Xoshiro(7 + 1000k), N, R0, r); end
    d = linelist(s, N)
    res = fit_one(Xoshiro(11), d, mdl)
    line = (@sprintf("(4) N=%-6d R0=%.1f r=%d %s: K=%d T=%d | %.0f s | β %.3f [%.3f, %.3f] (true %.3f) | IP %.2f [%.2f, %.2f] | R0 %.2f [%.2f, %.2f] | R̂ %.3f ESS %.0f div %d",
            N, R0, r, mdl, d.K, d.nsteps, res.time, res.β..., R0 / WBAR, res.ip..., res.R0..., res.rhat, res.ess, res.div))
    lock(() -> say(line), lk)
end
close(OUT)
