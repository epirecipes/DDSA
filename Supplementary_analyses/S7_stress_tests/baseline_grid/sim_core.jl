# Supplement S7, baseline grid: DDSA fitted to replicated outbreaks simulated from a stochastic model, so that the study
# tests the mean-field approximation as well as the sampler.
#
# Simulator (chain binomial mirroring the GLCT map of the draft):
#   population: n susceptibles (code variable N) + M = 1 initial infective, ρ = M/(n+M); the Binomial final-size term
#   counts only the n (a slight variant of the main-text convention, in which N = n + M; supplement S7 footnote);
#   new infections on step k ~ Binomial(S_k, 1 − exp(−β I_k Δt / (N+M)))   (frequency dependence, as the map's
#   p_inf = 1 − exp(−β I Δt) with I a fraction of the whole population);
#   each infective passes through r geometric stages, progressing with probability g = r/W̄ per step. Stage
#   progressions are independent across individuals, so drawing each individual's total duration W ~ NegBin(r, g)
#   (total trials) at infection is exactly equivalent to the stage-wise Binomial(I_k^(s), g) updates; it also gives the
#   line list directly. Timing as in the map: an individual infected on step k is infectious on steps k+1, …, k+W
#   and its recorded interval is δ = W.
# Fitting: the notebook's Scenario A models (full line list) with the Table S4 priors:
#   DPH (correctly specified): β ~ U(1e-4, 2.5), ρ ~ Beta(1, N−1), p ~ Beta(1, 1), r summed over 1..min(10, min δ);
#   geometric (misspecified when r = 4): r ≡ 1, same priors.
using Random, Statistics, Distributions, LinearAlgebra, Printf
using Turing, MCMCChains, MCMCDiagnosticTools
using StatsFuns: binomlogpdf
using SpecialFunctions: loggamma
const SA   = normpath(joinpath(@__DIR__, "..", ".."))
const ROOT = let d = @__DIR__   # repository root: nearest parent folder that holds the shared modules
    while !isfile(joinpath(d, "IntegerParameters.jl")) && dirname(d) != d; d = dirname(d); end; d
end
isdefined(Main, :PhaseTypeDistributions) || include(joinpath(ROOT, "PhaseTypeDistributions.jl"))
using .PhaseTypeDistributions
import Statistics: mean, var, std, median, quantile
include(joinpath(ROOT, "IntegerParameters.jl")); using .IntegerParameters
const OUTDIR = mkpath(joinpath(@__DIR__, get(ENV, "S7_OUT", "out")))

const WBAR = 4 / (1 - exp(-1))     # 6.328 d: discrete mean of the Erlang-4 DPH of the draft (μ = 4 d, Δt = 1 d)
const M_SEED = 1
const β_HI = 2.5
const R_MAX = 10

# ------------------------------------------------------------------ simulators
"""Chain-binomial outbreak. Returns (infday, recday, K, T, seed_delta) for the K new infections (+ the seed's interval)."""
function sim_chainbinomial(rng, N, R0, r; W̄ = WBAR, M = M_SEED, Tmax = 5000)
    β = R0 / W̄; g = r / W̄; Npop = N + M
    durdist = NegativeBinomial(r, g)               # failures before the r-th success; total trials = r + failures
    W() = r + rand(rng, durdist)
    rem = zeros(Int, Tmax + 400)                   # removals scheduled at the end of step j
    seedW = [W() for _ in 1:M]; for w in seedW; rem[w] += 1; end
    S = N; I = M
    infday = Int[]; recday = Int[]
    k = 0
    while I > 0 && k < Tmax
        k += 1
        pinf = 1 - exp(-β * I / Npop)
        n = rand(rng, Binomial(S, pinf))
        S -= n; I -= rem[k]
        for _ in 1:n
            w = W(); push!(infday, k); push!(recday, k + w); rem[k + w] += 1
        end
        I += n
    end
    return (infday = infday, recday = recday, K = length(infday), T = maximum(vcat(recday, seedW)), seed_delta = seedW)
end

"""Continuous-time Gillespie SIR with Erlang-r infectious period (stage rate r/μ), frequency-dependent infection rate
β_c S I / (N+M), μ = W̄ and R0 = β_c μ. Event times are floored to days; returns the same line-list format."""
function sim_gillespie(rng, N, R0, r; μ = WBAR, M = M_SEED)
    βc = R0 / μ; λ = r / μ; Npop = N + M
    stage = Int[]; tinf = Float64[]; who = Int[]          # infectives: current stage, infection time, index
    S = N; t = 0.0
    inf_t = Float64[]; rec_t = Float64[]; seed_rec = Float64[]
    for _ in 1:M; push!(stage, 1); push!(tinf, 0.0); push!(who, 0); end
    while !isempty(stage)
        I = length(stage)
        a_inf = βc * S * I / Npop; a_prog = λ * I; a = a_inf + a_prog
        t += randexp(rng) / a
        if rand(rng) * a < a_inf
            S -= 1; push!(inf_t, t); push!(rec_t, NaN)
            push!(stage, 1); push!(tinf, t); push!(who, length(inf_t))
        else
            j = rand(rng, 1:I)
            stage[j] += 1
            if stage[j] > r
                id = who[j]; id == 0 ? push!(seed_rec, t) : (rec_t[id] = t)
                deleteat!(stage, j); deleteat!(tinf, j); deleteat!(who, j)
            end
        end
    end
    # day k = floor(t) + 1 (day 1 covers [0, 1)), interval δ = recovery day − infection day, at least 1
    infday = floor.(Int, inf_t) .+ 1; recday = floor.(Int, rec_t) .+ 1
    δ = max.(1, recday .- infday)
    recday = infday .+ δ
    seedδ = max.(1, floor.(Int, seed_rec) .+ 1 .- 0)
    return (infday = infday, recday = recday, K = length(infday), T = maximum(vcat(recday, seedδ)), seed_delta = seedδ)
end

"""Line-list summaries used by the likelihood."""
function linelist(sim, N)
    T = sim.T
    th = zeros(Int, T); for d in sim.infday; th[d] += 1; end
    delta = vcat(sim.seed_delta, sim.recday .- sim.infday)
    du = sort(unique(delta)); dn = [count(==(x), delta) for x in du]
    mc = loggamma(sim.K + 1.0) - sum(loggamma.(th .+ 1.0))
    logCK = loggamma(N + 1.0) - loggamma(sim.K + 1.0) - loggamma(N - sim.K + 1.0)
    return (N = N, M = M_SEED, K = sim.K, th = th, nsteps = T, delta = delta, du = du, dn = dn, mc = mc, logCK = logCK)
end

# ------------------------------------------------------------------ likelihood (notebook dph_full_loglik, type-stable map)
"""Type-stable equivalent of the notebook's simulate_discrete_glct: returns τ and the normalised f (checked in sim_check.jl)."""
function glct_tau_f(Tm, α, β, ρ, nsteps)
    T = promote_type(eltype(Tm), typeof(β), typeof(ρ))
    S = one(T) - ρ; x = α .* ρ
    f = Vector{T}(undef, nsteps)
    S0 = S
    for t in 1:nsteps
        I = sum(x)
        p = one(T) - exp(-β * I)
        p = ifelse(isnan(p), zero(T), clamp(p, zero(T), one(T)))       # notebook rate_to_proportion guards
        ni = p * S
        x = Tm' * x .+ α .* ni
        S -= ni
        f[t] = ni
    end
    τ = S0 - S
    f = max.(f, zero(T)); sF = sum(f)
    f = sF > 0 ? f ./ sF : fill(one(T) / nsteps, nsteps)
    return τ, f
end
function safe_normalise(f::AbstractVector{T}, n::Int) where T
    f_pos = ifelse.(isnan.(f) .| .!isfinite.(f), zero(T), max.(f, zero(T)))
    sF = sum(f_pos)
    (sF > T(1e-15) && isfinite(sF)) ? f_pos ./ sF : fill(one(T)/n, n)
end
function full_loglik_guarded(r, β, ρ, p, d)   # notebook version (numerical guards); used only in the checks
    d.du[1] < r && return -Inf
    dph = PhaseTypeDistributions.dph_negative_binomial(r, p)
    τ, f = glct_tau_f(dph.S, dph.α, β, ρ, d.nsteps)
    τv = clamp(τ, 0.001, 0.999); fs = safe_normalise(f, d.nsteps)
    ll = binomlogpdf(d.N, τv, d.K)
    if d.K > 0
        (all(isfinite, fs) && abs(sum(fs) - 1) < 0.01) || return -Inf
        ll += logpdf(Multinomial(d.K, fs), d.th)
    end
    return ll + sum(n * logpdf(dph, x) for (x, n) in zip(d.du, d.dn))      # grouped by unique interval: same sum
end

# DPH model. The Table S4 prior (r uniform, p | r ~ Beta(1, 1)) is kept exactly, but sampled through ν = 1/(mean IP):
# ν ~ U(0, 1) as base measure, p = r ν, and each r carries the prior density of ν given r, r·1{rν < 1}. Sampling p
# directly makes the r-summed posterior multimodal in p (one mode per r, at p ≈ r/W̄), and chains started from the prior
# stay in those modes when there are many recovery intervals; with ν the mean IP has the same meaning for every r.
# ------------------------------------------------------------------ FINAL likelihood (log space, no flattening guards)
"""Log-space GLCT map: returns log f (normalised infection-time distribution), log τ and log(1 − τ), exactly.
log S_k = log S_0 − Σ_{j<k} λ_j with λ_j = β I_j; log new_k = log S_k + log(−expm1(−λ_k));
log τ = log S_0 + log(−expm1(−Σλ)); 1 − τ = ρ + S_T. No clamps, floors or uniform fallbacks."""
function glct_log(Tm, α, β, ρ, nsteps)
    T = promote_type(eltype(Tm), typeof(β), typeof(ρ))
    logS0 = log1p(-ρ); logS = logS0; x = α .* ρ; cumλ = zero(T)
    lognew = Vector{T}(undef, nsteps)
    for t in 1:nsteps
        λ = β * sum(x)
        lognew[t] = logS + log(-expm1(-λ))
        x = Tm' * x .+ α .* exp(lognew[t])
        logS -= λ; cumλ += λ
    end
    logτ = logS0 + log(-expm1(-cumλ))
    log1mτ = logaddexp(log(ρ), logS)
    return lognew .- logτ, logτ, log1mτ
end
logaddexp(a, b) = (m = max(a, b); isfinite(m) ? m + log(exp(a - m) + exp(b - m)) : m)

"""Asymptotic variance (per N) of the final size of a major outbreak (Andersson & Britton 2000, Thm 4.2; Ball 1986):
τ(1−τ)(1 + c²R0²(1−τ)) / (1 − R0(1−τ))², with c² the squared CV of the infectious period."""
fs_var(τ, R0, c2) = τ * (1 - τ) * (1 + c2 * R0^2 * (1 - τ)) / (1 - R0 * (1 - τ))^2

"""FINAL full-data log-likelihood. fsize = :binom (model of the paper), :none (no final-size term) or :normal
(K ~ Normal(Nτ, N σ²_Z) with the corrected final-size variance)."""
function full_loglik(r, β, ρ, p, d; fsize = :binom)
    d.du[1] < r && return -Inf
    (0 < p < 1) || return -Inf          # support of p (extreme NUTS proposals can underflow ν or p to 0)
    dph = PhaseTypeDistributions.dph_negative_binomial(r, p)
    logf, logτ, log1mτ = glct_log(dph.S, dph.α, β, ρ, d.nsteps)
    ll = d.mc                                                          # Multinomial constant
    @inbounds for k in 1:d.nsteps; d.th[k] > 0 && (ll += d.th[k] * logf[k]); end
    if fsize === :binom
        ll += d.logCK + d.K * logτ + (d.N - d.K) * log1mτ
    elseif fsize === :normal
        τ = exp(logτ); R0 = β * r / p; v = d.N * fs_var(τ, R0, (1 - p) / r)
        ll += -0.5 * log(2π * v) - (d.K - d.N * τ)^2 / (2v)
    end
    return ll + sum(n * logpdf(dph, x) for (x, n) in zip(d.du, d.dn))
end

@model function dph_full_sim(d, rmax; fsize = :binom, guarded = false)
    β ~ Uniform(1e-4, β_HI); ρ ~ Beta(1.0, d.N - 1); ν ~ Uniform(0.0, 1.0)
    logmarg, lps = marginal_r_loglik(1:rmax) do r
        r * ν < 1 || return -Inf
        log(r) + (guarded ? full_loglik_guarded(r, β, ρ, r * ν, d) : full_loglik(r, β, ρ, r * ν, d; fsize))
    end
    Turing.@addlogprob! logmarg
    return lps
end
@model function geo_full_sim(d; fsize = :binom, guarded = false)
    β ~ Uniform(1e-4, β_HI); ρ ~ Beta(1.0, d.N - 1); nb_p ~ Beta(1.0, 1.0)
    Turing.@addlogprob! (guarded ? full_loglik_guarded(1, β, ρ, nb_p, d) : full_loglik(1, β, ρ, nb_p, d; fsize))
end

# ------------------------------------------------------------------ one fit → summary row
qs(v) = (median(v), quantile(v, 0.025), quantile(v, 0.975))
function fit_one(rng, d, model_name; n_draws = 1000, n_adapt = 1000, n_chains = 4, init = :data, fsize = :binom, guarded = false)
    rmax = min(R_MAX, d.du[1])
    m = model_name === :dph ? dph_full_sim(d, rmax; fsize, guarded) : geo_full_sim(d; fsize, guarded)
    # Data-informed, jittered starts (chains started from the prior can stall where the notebook's numerical guards make
    # the likelihood flat, τ ≈ 0): ν from the mean recorded interval, β from the final-size relation, ρ = 1/(N+1).
    z = clamp(d.K / d.N, 0.01, 0.99); R0h = -log(1 - z) / z; ipd = sum(d.du .* d.dn) / sum(d.dn)
    inits = map(1:n_chains) do _
        j() = exp(0.1 * randn(rng))
        ν0 = clamp(j() / ipd, 1e-3, 0.99 / (model_name === :dph ? rmax : 1)); β0 = clamp(j() * R0h / ipd, 2e-4, 2.4)
        ρ0 = clamp(j() / (d.N + 1), 1e-7, 0.5)
        model_name === :dph ? InitFromParams((β = β0, ρ = ρ0, ν = ν0)) : InitFromParams((β = β0, ρ = ρ0, nb_p = ν0))
    end
    init === :prior && (inits = fill(InitFromPrior(), n_chains))
    t = @elapsed ch = sample(rng, m, NUTS(n_adapt, 0.8; max_depth = 12, Δ_max = 1000.0), MCMCSerial(), n_draws, n_chains;
                             progress = false, chain_type = MCMCChains.Chains, initial_params = inits)
    lpc = vec(mean(Array(ch[:logjoint]), dims = 1))                    # per-chain mean log density
    lpgap = maximum(mean(lpc[setdiff(1:n_chains, c)]) - lpc[c] for c in 1:n_chains)
    β = vec(ch[:β])
    if model_name === :dph
        lps = returned(m, ch)
        rd = vec([sample_index(rng, l) for l in lps])
        ip = 1 ./ vec(ch[:ν]); rmed = median(rd)
        s = summarystats(ch[[:β, :ρ, :ν]])
    else
        ip = 1 ./ vec(ch[:nb_p]); rmed = 1.0
        s = summarystats(ch[[:β, :ρ, :nb_p]])
    end
    R0 = β .* ip
    return (time = t, β = qs(β), ip = qs(ip), R0 = qs(R0), rmed = rmed,
            rhat = maximum(s[:, :rhat]), ess = minimum(s[:, :ess_bulk]), div = Int(sum(vec(ch[:numerical_error]))), lpgap = lpgap)
end
