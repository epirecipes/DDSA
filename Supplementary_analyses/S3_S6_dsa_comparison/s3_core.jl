# Supplement S3–S6: DDSA against continuous-time DSA on simulated outbreaks (shared code).
#
# Data: continuous-time stochastic SIR outbreaks (exact event times), the design of KhudaBukhsh et al. (2020, §3.4):
#   n susceptibles + m = round(ρ_S n) initial infectives (ρ_S = 0.05, initial infected fraction relative to susceptibles),
#   infection hazard β S I / (n + m), infectious periods drawn at infection from any distribution (exponential here).
# Each outbreak is observed up to a cut-off T (near the peak or near the end of the mean-field epidemic) in three forms:
#   exact times (continuous-time DSA as in the original paper), daily records (what a line list holds; DDSA), and daily
#   records with each event spread uniformly within its day (continuous-time DSA applied to daily line lists).
# Models (same priors and NUTS settings for the Bayesian arms; the DDSA model is the one of the main text, with a right-censoring term):
#   ddsa      discrete map on steps of Δt days, geometric infectious period, Multinomial + Binomial(N, τ) + recovery terms
#   dsa       our implementation of continuous-time DSA, SDS likelihood eq. (3.3): (n − K) log s_T + Σ log(β s(t_i) i(t_i)) + exponential terms
#   exact     complete-data (Doob–Gillespie) likelihood with n and the full trajectory known: conjugate Gamma posteriors
using Random, Statistics, Distributions, LinearAlgebra, Printf
using Turing, MCMCChains, MCMCDiagnosticTools
using SpecialFunctions: loggamma
const SA   = normpath(joinpath(@__DIR__, ".."))
const ROOT = let d = @__DIR__   # repository root: nearest parent folder that holds the shared modules
    while !isfile(joinpath(d, "IntegerParameters.jl")) && dirname(d) != d; d = dirname(d); end; d
end
isdefined(Main, :IntegerParameters) || (include(joinpath(ROOT, "IntegerParameters.jl")); using .IntegerParameters)
const OUTDIR = mkpath(joinpath(@__DIR__, get(ENV, "S3_OUT", "out")))

# ------------------------------------------------------------------ priors (shared by every model)
const β_LO, β_HI = 1e-4, 2.5          # per day
const μ_LO, μ_HI = 1.0, 30.0          # mean infectious period, days
const ρ_A, ρ_B   = 1.0, 10.0          # initial infected fraction of the whole population

# ------------------------------------------------------------------ simulator
"""Continuous-time SIR outbreak with n susceptibles and m initial infectives (infected at t = 0), infection hazard
β S I/(n+m) and infectious periods drawn from `ipd` at infection. Runs to extinction. Exact event-driven simulation: the
infection process is memoryless between events, so after each event the next infection time is redrawn."""
function sim_outbreak(rng, n, m, β, ipd)
    Npop = n + m; S = n; t = 0.0
    pend = sort!([rand(rng, ipd) for _ in 1:m])           # pending recovery times (sorted)
    seed_rec = copy(pend)
    inf_t = Float64[]; rec_t = Float64[]
    while !isempty(pend)
        I = length(pend)
        a = β * S * I / Npop
        tn = a > 0 ? t + randexp(rng) / a : Inf
        if tn < pend[1]
            t = tn; S -= 1
            r = t + rand(rng, ipd)
            push!(inf_t, t); push!(rec_t, r)
            insert!(pend, searchsortedfirst(pend, r), r)
        else
            t = popfirst!(pend)
        end
    end
    return (n = n, m = m, inf_t = inf_t, rec_t = rec_t, seed_rec = seed_rec)
end

"""Mean-field SIR (exponential infectious period, mean μ) in fractions of the whole population: incidence curve on a
fine grid; returns the time of peak incidence and the time by which 99% of the final size is reached."""
function meanfield_cutoffs(R0, μ, ρtot; h = 0.001, tmax = 2000.0)
    β = R0 / μ; γ = 1 / μ; S = 1 - ρtot; I = ρtot; C = 0.0
    ts = Float64[]; inc = Float64[]; cs = Float64[]; t = 0.0
    while t < tmax && (I > 1e-9 || t < 1)
        f(S, I) = (-β * S * I, β * S * I - γ * I)
        k1 = f(S, I); k2 = f(S + h/2*k1[1], I + h/2*k1[2]); k3 = f(S + h/2*k2[1], I + h/2*k2[2]); k4 = f(S + h*k3[1], I + h*k3[2])
        dS = h/6*(k1[1] + 2k2[1] + 2k3[1] + k4[1]); dI = h/6*(k1[2] + 2k2[2] + 2k3[2] + k4[2])
        S += dS; I += dI; C -= dS; t += h
        push!(ts, t); push!(inc, β * S * I); push!(cs, C)
    end
    tpeak = ts[argmax(inc)]; t99 = ts[findfirst(>=(0.99 * cs[end]), cs)]
    return (tpeak = tpeak, t99 = t99, final = cs[end] / (1 - ρtot))
end

# ------------------------------------------------------------------ observation formats
"""Exact times up to the cut-off T: infection times t_i < T (sorted), complete intervals w_i (recovery before T),
censored exposures, and the seeds' recoveries."""
function obs_exact(sim, T)
    keep = findall(<(T), sim.inf_t)
    ti = sim.inf_t[keep]; tr = sim.rec_t[keep]
    comp = tr .< T
    nrec = count(comp) + count(<(T), sim.seed_rec)
    expo = sum(tr[comp] .- ti[comp]; init = 0.0) + sum(T .- ti[.!comp]; init = 0.0) +
           sum(min.(sim.seed_rec, T); init = 0.0)
    return (n = sim.n, m = sim.m, T = float(T), ti = ti, K = length(ti), nrec = nrec, expo = expo)
end

"""Records at resolution Δt (days) up to T: infection step k = ⌊t/Δt⌋ + 1, interval δ = max(1, recovery step − k)
(seeds are infected at step 0), right-censored at the last step if recovery falls at or after T."""
function obs_steps(sim, T; Δt = 1.0)
    nsteps = round(Int, T / Δt)
    th = zeros(Int, nsteps); dvals = Int[]; cvals = Int[]
    for (t, r) in zip(sim.inf_t, sim.rec_t)
        t < T || continue
        k = floor(Int, t / Δt) + 1; th[k] += 1
        r < T ? push!(dvals, max(1, floor(Int, r / Δt) + 1 - k)) : push!(cvals, nsteps - k)
    end
    for r in sim.seed_rec
        r < T ? push!(dvals, floor(Int, r / Δt) + 1) : push!(cvals, nsteps)
    end
    K = sum(th); Ntot = sim.n + sim.m
    du = sort(unique(dvals)); dn = [count(==(x), dvals) for x in du]
    cu = sort(unique(cvals)); cn = [count(==(x), cvals) for x in cu]
    mc = loggamma(K + 1.0) - sum(loggamma.(th .+ 1.0))
    logCK = loggamma(Ntot + 1.0) - loggamma(K + 1.0) - loggamma(Ntot - K + 1.0)
    return (n = sim.n, m = sim.m, Ntot = Ntot, Δt = Δt, nsteps = nsteps, th = th, K = K, du = du, dn = dn, cu = cu, cn = cn,
            mc = mc, logCK = logCK, meanδ = isempty(dvals) ? NaN : mean(dvals) * Δt)
end

"""Daily records turned back into times by spreading each event uniformly within its day (as done for daily data in
Di Lauro et al. 2022); an infection and a recovery on the same day are ordered. Censoring as in the exact data."""
function obs_jitter(rng, sim, T; Δt = 1.0)
    ti = Float64[]; nrec = 0; expo = 0.0
    for (t, r) in zip(sim.inf_t, sim.rec_t)
        t < T || continue
        k = floor(Int, t / Δt)
        if r < T
            kr = floor(Int, r / Δt)
            if kr == k
                u1, u2 = minmax(rand(rng), rand(rng)); a = (k + u1) * Δt; b = (k + u2) * Δt
            else
                a = (k + rand(rng)) * Δt; b = (kr + rand(rng)) * Δt
            end
            push!(ti, a); nrec += 1; expo += b - a
        else
            a = (k + rand(rng)) * Δt; push!(ti, a); expo += T - a
        end
    end
    for r in sim.seed_rec
        if r < T; nrec += 1; expo += (floor(r / Δt) + rand(rng)) * Δt; else; expo += T; end
    end
    sort!(ti)
    return (n = sim.n, m = sim.m, T = float(T), ti = ti, K = length(ti), nrec = nrec, expo = expo)
end

# ------------------------------------------------------------------ likelihoods
logaddexp(a, b) = (m = max(a, b); isfinite(m) ? m + log(exp(a - m) + exp(b - m)) : m)

"""DDSA log-likelihood (geometric infectious period), on the log scale. Map in steps of Δt days with per-step
infection probability 1 − exp(−β I Δt) and recovery probability p = Δt/μ (so the mean infectious period is μ days):
Multinomial(K; f) for the infection steps, Binomial(N, τ) for the final size (N = n + m, τ = S_0 − S_T, as in the
notebooks), 1-indexed geometric PMF for complete intervals and its survivor function for right-censored ones."""
function ddsa_loglik(β, μ, ρ, d)
    p = d.Δt / μ
    (0 < p < 1 && 0 < ρ < 1) || return -Inf
    T_ = promote_type(typeof(β), typeof(μ), typeof(ρ))
    logS0 = log1p(-ρ); logS = logS0; I = T_(ρ); cumλ = zero(T_)
    acc = zero(T_)
    @inbounds for k in 1:d.nsteps
        λ = β * I * d.Δt
        lognew = logS + log(-expm1(-λ))
        d.th[k] > 0 && (acc += d.th[k] * lognew)
        I = I * (1 - p) + exp(lognew)
        logS -= λ; cumλ += λ
    end
    logτ = logS0 + log(-expm1(-cumλ)); log1mτ = logaddexp(log(ρ), logS)
    ll = d.mc + (acc - d.K * logτ) + d.logCK + d.K * logτ + (d.Ntot - d.K) * log1mτ
    l1p = log1p(-p); lp = log(p)
    for (x, c) in zip(d.du, d.dn); ll += c * (lp + (x - 1) * l1p); end
    for (x, c) in zip(d.cu, d.cn); ll += c * x * l1p; end
    return ll
end

"""Continuous-time DSA log-likelihood, SDS eq. (3.3) with n known (KhudaBukhsh et al. 2020): mean-field SIR in fractions
of the whole population (S_0 = 1 − ρ, I_0 = ρ), integrated in log coordinates (u = log S, v = log I) by RK4 with steps
of at most h days, stepping exactly to every infection time. Per susceptible, the infection density is β S(t) I(t)/S_0
and the survival S(T)/S_0; recoveries are exponential with rate 1/μ, with censoring."""
dsa_loglik(β, μ, ρ, d; h = 0.02) = (r = dsa_parts(β, μ, ρ, d; h); r.ll + (d.n - d.K) * r.logsT)

"""Parts of the continuous-time DSA log-likelihood: the infection-density and recovery terms (ll), and the
per-susceptible log survival to T, log(s_T/s_0) (logsT), which the final-size factor (n − K) log(s_T/s_0) uses."""
function dsa_parts(β, μ, ρ, d; h = 0.02)
    T_ = promote_type(typeof(β), typeof(μ), typeof(ρ))
    (0 < ρ < 1 && μ > 0) || return (ll = T_(-Inf), logsT = zero(T_))
    γ = 1 / μ
    logS0 = log1p(-ρ); u = T_(logS0); v = T_(log(ρ)); t = 0.0
    fu(u, v) = -β * exp(v); fv(u, v) = β * exp(u) - γ
    function advance(u, v, t, t1)
        while t < t1
            s = min(h, t1 - t)
            a1 = fu(u, v);                      b1 = fv(u, v)
            a2 = fu(u + s/2*a1, v + s/2*b1);    b2 = fv(u + s/2*a1, v + s/2*b1)
            a3 = fu(u + s/2*a2, v + s/2*b2);    b3 = fv(u + s/2*a2, v + s/2*b2)
            a4 = fu(u + s*a3, v + s*b3);        b4 = fv(u + s*a3, v + s*b3)
            u += s/6*(a1 + 2a2 + 2a3 + a4); v += s/6*(b1 + 2b2 + 2b3 + b4); t += s
        end
        return u, v, t
    end
    ll = zero(T_); lβ = log(β)
    for ti in d.ti
        u, v, t = advance(u, v, t, ti)
        ll += lβ + u + v - logS0
    end
    u, v, t = advance(u, v, t, d.T)
    return (ll = ll + d.nrec * log(γ) - γ * d.expo, logsT = u - logS0)
end

# ------------------------------------------------------------------ Turing models
@model function ddsa_model(d)
    β ~ Uniform(β_LO, β_HI); μ ~ Uniform(μ_LO, μ_HI); ρ ~ Beta(ρ_A, ρ_B)
    Turing.@addlogprob! ddsa_loglik(β, μ, ρ, d)
end
@model function dsa_model(d)
    β ~ Uniform(β_LO, β_HI); μ ~ Uniform(μ_LO, μ_HI); ρ ~ Beta(ρ_A, ρ_B)
    Turing.@addlogprob! dsa_loglik(β, μ, ρ, d)
end

# ------------------------------------------------------------------ fitting and summaries
summ(v) = (mean(v), std(v), median(v), quantile(v, 0.025), quantile(v, 0.975))

"""Data-based, jittered starts (as in the main text): μ from the mean complete interval, ρ from the observed number of
initial infectives, β from the final-size relation at the end cut-off (R0 = 2 otherwise)."""
function init_values(rng, d, meanint, cut; n_chains = 4)
    z = clamp(d.K / d.n, 0.01, 0.99)
    R0h = cut === :end ? clamp(-log(1 - z) / z, 1.05, 6.0) : 2.0
    μh = isfinite(meanint) ? clamp(meanint, 1.5, 25.0) : 5.0
    map(1:n_chains) do _
        j() = exp(0.1 * randn(rng))
        (β = clamp(j() * R0h / μh, 2e-4, 2.4), μ = clamp(j() * μh, 1.05, 29.0), ρ = clamp(j() * d.m / (d.n + d.m), 1e-6, 0.5))
    end
end
inits_for(rng, d, meanint, cut; n_chains = 4) = InitFromParams.(init_values(rng, d, meanint, cut; n_chains))

"""Fit `model` (ddsa_model or dsa_model) to data d. Returns posterior summaries (mean, sd, median, 2.5%, 97.5%) of β,
mean IP μ, R0 = β μ and ρ, with run time and diagnostics."""
function fit_bayes(rng, m, d, meanint, cut; n_draws = 1000, n_adapt = 1000, n_chains = 4)
    inits = inits_for(rng, d, meanint, cut; n_chains)
    t = @elapsed ch = sample(rng, m, NUTS(n_adapt, 0.8; max_depth = 12, Δ_max = 1000.0), MCMCSerial(), n_draws, n_chains;
                             progress = false, chain_type = MCMCChains.Chains, initial_params = inits)
    β = vec(ch[:β]); μ = vec(ch[:μ]); ρ = vec(ch[:ρ])
    s = summarystats(ch[[:β, :μ, :ρ]])
    arr = permutedims(cat([Array(ch[:, [:β, :μ, :ρ], c]) for c in 1:n_chains]...; dims = 3), (1, 3, 2))   # draws × chains × params
    ess_tail = minimum(MCMCDiagnosticTools.ess(arr; kind = :tail))
    lpc = vec(mean(Array(ch[:logjoint]), dims = 1))
    lpgap = maximum(mean(lpc[setdiff(1:n_chains, c)]) - lpc[c] for c in 1:n_chains)
    return (time = t, β = summ(β), μ = summ(μ), R0 = summ(β .* μ), ρ = summ(ρ),
            rhat = maximum(s[:, :rhat]), ess = minimum(s[:, :ess_bulk]), ess_tail = ess_tail,
            div = Int(sum(vec(ch[:numerical_error]))), lpgap = lpgap)
end

"""Complete-data benchmark (SDS Methods 1–2): with n and the full trajectory on [0, T] known, β | X ~ Gamma(z_I + a,
∫ S I/(n+m) dt + b) and γ | X ~ Gamma(z_R + a, ∫ I dt + b), a = b = 0.001."""
function fit_exact(rng, sim, T; ndraw = 4000, a = 0.001, b = 0.001)
    Npop = sim.n + sim.m
    ev = vcat([(t, +1) for t in sim.inf_t if t < T], [(r, -1) for r in vcat(sim.rec_t, sim.seed_rec) if r < T])
    sort!(ev, by = first)
    S = sim.n; I = sim.m; t = 0.0; ∫SI = 0.0; ∫I = 0.0; zI = 0; zR = 0
    for (te, ty) in ev
        ∫SI += S * I * (te - t) / Npop; ∫I += I * (te - t); t = te
        ty == 1 ? (S -= 1; I += 1; zI += 1) : (I -= 1; zR += 1)
    end
    ∫SI += S * I * (T - t) / Npop; ∫I += I * (T - t)
    β = rand(rng, Gamma(zI + a, 1 / (∫SI + b)), ndraw); γ = rand(rng, Gamma(zR + a, 1 / (∫I + b)), ndraw)
    μ = 1 ./ γ; ρv = fill(sim.m / Npop, ndraw)
    return (time = 0.0, β = summ(β), μ = summ(μ), R0 = summ(β .* μ), ρ = summ(ρv),
            rhat = 1.0, ess = float(ndraw), ess_tail = float(ndraw), div = 0, lpgap = 0.0)
end
