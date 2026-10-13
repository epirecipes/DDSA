# Supplement S7: stress tests of DDSA alone on synthetic data. One baseline (chain-binomial outbreak, n = 1,000 susceptibles
# (code variable N) plus one seed, R0 = 2, Erlang-4-type infectious period with mean W̄ = 6.33 d, full line list, N known;
# the S7 baseline-grid generator of ../S7_stress_tests/baseline_grid/sim_core.jl), changing one factor at a time. Each dataset
# is fitted with the main-text NegBin DPH model (r summed out, priors of Table S4, sampled through ν = 1/mean IP,
# data-based starts), extended where a factor needs it: right-censoring or a truncation term (window cut), and N summed
# out (unknown N).
# Detection: posterior predictive p-values of the daily incidence, the interval distribution and the final size.
include(joinpath(@__DIR__, "baseline_grid", "sim_core.jl"))   # simulators, glct_log, DPH, priors
const OUT7 = mkpath(joinpath(@__DIR__, get(ENV, "S7S_OUT", "out")))
const N7, R07, r7 = 1000, 2.0, 4

# ------------------------------------------------------------------ general chain-binomial generator
"""Chain-binomial outbreak with individual heterogeneity. With all options at their defaults it makes the same random
draws, in the same order, as `sim_chainbinomial` (checked in stress_check.jl). Options:
  βfun(k)     transmission rate on day k (default constant R0/W̄)
  susc        vector of relative susceptibilities of the N susceptibles (default all 1)
  kdisp       dispersion k of individual infectiousness ν ~ Gamma(k, 1/k) (default none: ν ≡ 1)
  imports     days on which one susceptible is infected from outside (default none)
  late_seeds  (day, group) pairs: within-group mixing for out-of-phase subpopulations (see sim_subpops)
Timing as in the map: a case infected on day k is infectious on days k+1, …, k+W and has interval δ = W."""
function sim_cb_general(rng, N, R0, r; W̄ = WBAR, M = M_SEED, Tmax = 5000, βfun = nothing, susc = nothing,
                        kdisp = nothing, imports = Int[])
    β0 = R0 / W̄; g = r / W̄; Npop = N + M
    durdist = NegativeBinomial(r, g); W() = r + rand(rng, durdist)
    ν() = kdisp === nothing ? 1.0 : rand(rng, Gamma(kdisp, 1 / kdisp))
    remΛ = zeros(Tmax + 400); remI = zeros(Int, Tmax + 400)    # infectiousness and cases leaving at the end of day j
    seedW = [W() for _ in 1:M]; seedν = [ν() for _ in 1:M]
    for (w, v) in zip(seedW, seedν); remΛ[w] += v; remI[w] += 1; end
    S = N; Λ = sum(seedν); I = M; alive = susc === nothing ? nothing : collect(1:N)
    infday = Int[]; recday = Int[]; k = 0; imp = Set(imports)
    while (I > 0 || any(>(k), imports)) && k < Tmax
        k += 1
        β = βfun === nothing ? β0 : βfun(k)
        Λ = max(Λ, 0.0)
        if susc === nothing
            n = rand(rng, Binomial(S, 1 - exp(-β * Λ / Npop)))
        else
            hit = [rand(rng) < 1 - exp(-susc[i] * β * Λ / Npop) for i in alive]
            n = count(hit); alive = alive[.!hit]
        end
        if k in imp && S - n > 0
            n += 1
            susc === nothing || deleteat!(alive, rand(rng, 1:length(alive)))
        end
        S -= n; Λ -= remΛ[k]; I -= remI[k]
        for _ in 1:n
            w = W(); v = ν(); push!(infday, k); push!(recday, k + w); remΛ[k + w] += v; remI[k + w] += 1; Λ += v; I += 1
        end
    end
    return (infday = infday, recday = recday, K = length(infday), T = maximum(vcat(recday, seedW)), seed_delta = seedW)
end

"""Five closed subpopulations of N/5 with within-group mixing only, seeded by one infective each on days
0, 15, 30, 45, 60. The first group's seed is the line list's initial infective; the later seeds appear as cases."""
function sim_subpops(rng, N, R0, r; G = 5, gap = 15)
    ng = N ÷ G; infday = Int[]; recday = Int[]; seed = Int[]
    for gi in 1:G
        s = sim_chainbinomial(rng, ng, R0, r)
        off = (gi - 1) * gap
        append!(infday, s.infday .+ off); append!(recday, s.recday .+ off)
        if gi == 1
            append!(seed, s.seed_delta)
        else
            for w in s.seed_delta; push!(infday, off); push!(recday, off + w); end   # later seeds: cases on their day
        end
    end
    infday = max.(infday, 1)
    return (infday = infday, recday = recday, K = length(infday), T = maximum(vcat(recday, seed)), seed_delta = seed)
end

# ------------------------------------------------------------------ line list with observation options
"""Build the likelihood summaries from a simulated outbreak.
  T        observation end (default: the end of the outbreak)
  thin     probability that a case is recorded (under-ascertainment; seeds kept)
  miss     probability that a recorded case's recovery date is missing (MCAR), or :long for duration-dependent
           missingness (60% of intervals above the median, about 30% overall)
  misrec   fraction of recorded intervals replaced by 1 day
  window   :censor (unrecovered cases right-censored), :drop (their intervals ignored), :trunc (ignored, and each
           complete interval conditioned on δ ≤ c, the follow-up from its infection day to T)"""
function stress_data(rng, s, N; T = s.T, thin = 1.0, miss = 0.0, misrec = 0.0, window = :censor)
    th = zeros(Int, T); comp = Int[]; cens = Int[]; trunc = Tuple{Int,Int}[]
    keep = [rand(rng) < thin for _ in s.infday]
    for (i, (k, rd)) in enumerate(zip(s.infday, s.recday))
        (keep[i] && k <= T) || continue
        th[k] += 1
        δ = rd - k
        if rd <= T
            push!(comp, δ); push!(trunc, (δ, T - k))
        elseif window === :censor
            push!(cens, T - k)
        end
    end
    for w in s.seed_delta
        w <= T ? (push!(comp, w); push!(trunc, (w, T))) : (window === :censor && push!(cens, T))
    end
    if miss === :long
        med = median(comp); obs = [!(δ > med && rand(rng) < 0.6) for δ in comp]
    else
        obs = [rand(rng) >= miss for _ in comp]
    end
    comp = comp[obs]; trunc = trunc[obs]
    if misrec > 0
        idx = randperm(rng, length(comp))[1:round(Int, misrec * length(comp))]
        comp[idx] .= 1; for i in idx; trunc[i] = (1, trunc[i][2]); end
    end
    K = sum(th)
    du = sort(unique(comp)); dn = [count(==(x), comp) for x in du]
    cu = sort(unique(cens)); cn = [count(==(x), cens) for x in cu]
    tu = sort(unique(trunc)); tn = [count(==(x), trunc) for x in tu]
    mc = loggamma(K + 1.0) - sum(loggamma.(th .+ 1.0))
    logCK = loggamma(N + 1.0) - loggamma(K + 1.0) - loggamma(N - K + 1.0)
    return (N = N, M = M_SEED, K = K, th = th, nsteps = T, du = du, dn = dn, cu = cu, cn = cn, tu = tu, tn = tn,
            window = window, mc = mc, logCK = logCK, meanδ = isempty(comp) ? WBAR : mean(comp))
end

# ------------------------------------------------------------------ likelihood and model
lchoose(n, k) = loggamma(n + 1.0) - loggamma(k + 1.0) - loggamma(n - k + 1.0)
"""log P(W = w) and log P(W > c) for the r-stage NegBin DPH (total trials)."""
nb_logpmf(w, r, lp, l1p) = w < r ? -Inf : lchoose(w - 1, r - 1) + r * lp + (w - r) * l1p
"""log P(W ≤ c) = log P(at least r successes in c trials), summed directly (stable when P(W ≤ c) is small)."""
function nb_logcdf(c, r, lp, l1p)
    c < r && return -Inf * lp
    t = [lchoose(c, j) + j * lp + (c - j) * l1p for j in r:c]
    m = maximum(t)
    return m + log(sum(exp.(t .- m)))
end
function nb_logsf(c, r, lp, l1p)
    c < r && return 0.0 * lp
    m = maximum(lchoose(c, j) + j * lp + (c - j) * l1p for j in 0:(r - 1))
    return m + log(sum(exp(lchoose(c, j) + j * lp + (c - j) * l1p - m) for j in 0:(r - 1)))
end

"""Main-text DDSA likelihood (log scale, glct_log of sim_core) with censoring/truncation terms; final size either
Binomial(N, τ) with N known (Nmode = :known) or summed over N (Nmode = :fraction, :count; the latter with the seed
prior I0 = ρN ~ Exp(1), which makes the prior on ρ depend on N)."""
function stress_loglik(r, β, ρ, p, d; Nmode = :known, Ngrid = nothing)
    (0 < p < 1 && 0 < ρ < 1) || return -Inf
    lp = log(p); l1p = log1p(-p)
    any(x -> x < r, d.du) && return -Inf
    dph = PhaseTypeDistributions.dph_negative_binomial(r, p)
    logf, logτ, log1mτ = glct_log(dph.S, dph.α, β, ρ, d.nsteps)
    ll = d.mc
    @inbounds for k in 1:d.nsteps; d.th[k] > 0 && (ll += d.th[k] * logf[k]); end
    if Nmode === :known
        ll += d.logCK + d.K * logτ + (d.N - d.K) * log1mτ
    elseif Nmode === :fraction
        ll += marginal_N_loglik(Ngrid, logτ, log1mτ)
    else                                                # count-scale seed prior: Σ_N π(N) C(N,K) τ^K (1−τ)^(N−K) N e^{−ρN}
        terms = [Ngrid.logprior[i] + Ngrid.logC[i] + (Ngrid.Ns[i] - Ngrid.K) * log1mτ + log(Ngrid.Ns[i]) - ρ * Ngrid.Ns[i]
                 for i in eachindex(Ngrid.Ns)]
        mx = maximum(terms); ll += d.K * logτ + mx + log(sum(exp.(terms .- mx)))
    end
    if d.window === :trunc
        for ((w, c), n) in zip(d.tu, d.tn)
            ll += n * (nb_logpmf(w, r, lp, l1p) - nb_logcdf(c, r, lp, l1p))
        end
    else
        for (w, n) in zip(d.du, d.dn); ll += n * nb_logpmf(w, r, lp, l1p); end
        for (c, n) in zip(d.cu, d.cn); ll += n * nb_logsf(c, r, lp, l1p); end
    end
    return ll
end

@model function stress_model(d, rmax; Nmode = :known, Ngrid = nothing)
    β ~ Uniform(1e-4, β_HI)
    if Nmode === :known
        ρ ~ Beta(1.0, d.N - 1)
    elseif Nmode === :fraction
        ρ ~ Beta(1.0, 10.0)
    else
        ρ ~ Uniform(0.0, 1.0)                           # base measure; the Exp(1) prior on ρN is inside the N sum
    end
    ν ~ Uniform(0.0, 1.0)
    logmarg, lps = marginal_r_loglik(1:rmax) do r
        r * ν < 1 || return -Inf
        log(r) + stress_loglik(r, β, ρ, r * ν, d; Nmode, Ngrid)
    end
    Turing.@addlogprob! logmarg
    return lps
end

# ------------------------------------------------------------------ fit + posterior predictive checks
"""Fit, then posterior predictive p-values from 200 draws: incidence (χ²-type discrepancy of the daily counts against
Multinomial(K, f)), intervals (the same for the complete intervals against the DPH PMF; not computed when cases are censored,
dropped or truncated, i.e. not at the observation-window levels), final size (two-sided, K against Binomial(N, τ))."""
function fit_stress(rng, d; Nmode = :known, Nmax = 10 * d.N, n_draws = 1000, n_adapt = 1000, n_chains = 4)
    rmax = isempty(d.du) ? R_MAX : min(R_MAX, d.du[1])
    Ngrid = Nmode === :known ? nothing : NGrid(d.K, d.K + 1, Nmax)
    m = stress_model(d, rmax; Nmode, Ngrid)
    z = clamp(d.K / d.N, 0.01, 0.99); R0h = -log(1 - z) / z; ipd = d.meanδ
    inits = map(1:n_chains) do _
        j() = exp(0.1 * randn(rng))
        ρ0 = Nmode === :known ? clamp(j() / (d.N + 1), 1e-7, 0.5) : clamp(j() / (d.N + 1), 1e-5, 0.5)
        InitFromParams((β = clamp(j() * R0h / ipd, 2e-4, 2.4), ρ = ρ0, ν = clamp(j() / ipd, 1e-3, 0.99 / rmax)))
    end
    t = @elapsed ch = sample(rng, m, NUTS(n_adapt, 0.8; max_depth = 12, Δ_max = 1000.0), MCMCSerial(), n_draws, n_chains;
                             progress = false, chain_type = MCMCChains.Chains, initial_params = inits)
    lps = returned(m, ch); rd = vec([sample_index(rng, l) for l in lps])
    β = vec(ch[:β]); ρ = vec(ch[:ρ]); ν = vec(ch[:ν]); ip = 1 ./ ν; R0 = β .* ip
    s = summarystats(ch[[:β, :ρ, :ν]])
    arr = permutedims(cat([Array(ch[:, [:β, :ρ, :ν], c]) for c in 1:n_chains]...; dims = 3), (1, 3, 2))
    # posterior predictive checks
    idx = rand(rng, eachindex(β), 200); pinc = 0; pdel = 0; pK = 0
    ncomp = sum(d.dn); Dδobs_ok = d.window === :censor && isempty(d.cu)
    for i in idx
        r = rd[i]; p = r * ν[i]
        dph = PhaseTypeDistributions.dph_negative_binomial(r, p)
        logf, logτ, _ = glct_log(dph.S, dph.α, β[i], ρ[i], d.nsteps)
        f = exp.(logf); f ./= sum(f)
        hrep = rand(rng, Multinomial(d.K, f))
        D(h) = sum((h[k] - d.K * f[k])^2 / (d.K * f[k] + 0.5) for k in eachindex(f))
        pinc += D(hrep) >= D(d.th)
        if Dδobs_ok && ncomp > 0
            xmax = max(4 * maximum(d.du), 60); lp = log(p); l1p = log1p(-p)
            q = [exp(nb_logpmf(x, r, lp, l1p)) for x in 1:xmax]; q ./= sum(q)
            obs = zeros(Int, xmax); for (x, n) in zip(d.du, d.dn); obs[min(x, xmax)] += n; end
            Dq(o) = sum((o[x] - ncomp * q[x])^2 / (ncomp * q[x] + 0.5) for x in 1:xmax)
            pdel += Dq(rand(rng, Multinomial(ncomp, q))) >= Dq(obs)
        end
        Nd = Nmode === :known ? d.N : sample_N(rng, Ngrid, exp(logτ))
        Krep = rand(rng, Binomial(Nd, exp(logτ))); pK += Krep >= d.K
    end
    pK = pK / 200; pK2 = min(1.0, 2 * min(pK, 1 - pK))
    return (time = t, β = qs(β), ip = qs(ip), R0 = qs(R0), rmed = median(rd), rmax = rmax,
            rhat = maximum(s[:, :rhat]), ess = minimum(s[:, :ess_bulk]), ess_tail = minimum(MCMCDiagnosticTools.ess(arr; kind = :tail)),
            div = Int(sum(vec(ch[:numerical_error]))), p_inc = pinc / 200, p_delta = Dδobs_ok ? pdel / 200 : NaN, p_K = pK2,
            sd_R0 = std(R0), sd_ip = std(ip))
end
