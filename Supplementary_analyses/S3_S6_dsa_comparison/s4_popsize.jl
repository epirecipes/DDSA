# Supplement S4: unknown population size. Same simulated outbreaks as S3 (s3_1_run.jl), with n treated as unknown.
#   ddsa_N      DDSA (daily records) with N summed out of Binomial(N, τ) over N ∈ {K+1, …, N_max = 5n} (NGrid, as in the
#               H1N1 and Ebola notebooks), sampled with NUTS; N drawn afterwards from its exact conditional.
#   dsa_sds     continuous-time DSA (exact times), our re-implementation of Algorithm 4.1 of KhudaBukhsh et al. (2020): the likelihood
#               conditional on n, eq. (3.3) without C(n, K), and n imputed each iteration as n = K + NegBin(K, τ_∞), with
#               τ_∞ from the final-size equation (2.3) solved by 50 fixed-point iterations; improper uniform prior on n.
#   dsa_window  as dsa_sds, with n imputed from the attack probability by the end of the observation window instead of
#               the final size, n = K + NegBin(K, q_T) with q_T = 1 − s_T/s_0, the window-matched counterpart used by the
#               estimator n̂_T = k_T/(1 − z_S(T)) of Di Lauro et al. (2022, eq. 3.12). The pair dsa_sds/dsa_window
#               isolates the window of the imputation; both omit C(n, K). dsa_window restricts n to ≤ N_max, the support
#               of ddsa_N and dsa_full (the dsa_sds draws stay far below N_max).
#   dsa_full    continuous-time DSA (exact times) with the complete Binomial final-size term, C(n, K) q^K (1 − q)^(n−K)
#               with q = 1 − s_T/s_0, and n summed out over the same support as ddsa_N; sampled with NUTS. Same data and
#               likelihood family as dsa_sds; it differs only in how n is treated, so the pair isolates the n step.
#               (A Gibbs version with n − K ~ NegBin(K + 1, q) targets the same posterior but mixes slowly along the
#               n–ρ–β ridge at the near-peak cut-off.)
# In dsa_sds, θ = (β, μ, ρ) is updated by adaptive random-walk Metropolis (Haario et al. 2001; proposal covariance
# adapted during burn-in only) on (log β, log μ, logit ρ), with the S3 priors, alternating with the n imputation.
#   S4_REPS=10  BUDGET=420  RUN=s4_main
# Usage (from the supplementary folder): REPS=5 RUN=s4_pilot julia --project=@. -t 10 S3_S6_dsa_comparison/s4_popsize.jl
include(joinpath(@__DIR__, "s3_core.jl"))
const REPS   = parse(Int, get(ENV, "S4_REPS", "10"))   # S4 ignores the queue's REPS (changed from 20 to 10 on 2026-10-08)
const BUDGET = parse(Float64, get(ENV, "BUDGET", "420"))
const RUN    = get(ENV, "RUN", "s4_main")
const MU = 5.0; const RHO_S = 0.05; const NS = (100, 1000, 10_000); const R0S = (1.5, 2.0, 4.0)
const NMAX_FACTOR = 5                        # N_max = 5n, for ddsa_N and dsa_full (changed from 10(n + m) on 2026-10-08)
const N_BURN, N_KEEP, N_CH = 5000, 10_000, 4
# dsa_window moves slowly along the n–ρ–β ridge: its chains are this many times longer (end and peak cut-off), thinned by
# the same factor so that every variant keeps N_KEEP draws per chain
const WINDOW_MULT      = parse(Int, get(ENV, "S4_WINDOW_MULT", "5"))
const WINDOW_MULT_PEAK = parse(Int, get(ENV, "S4_WINDOW_MULT_PEAK", "20"))

cutoffs(n, R0) = (mf = meanfield_cutoffs(R0, MU, round(Int, RHO_S * n) / (n + round(Int, RHO_S * n)));
                  (peak = ceil(mf.tpeak), stop = ceil(mf.t99)))
simulate(n, R0, rep) = sim_outbreak(Xoshiro(hash(("s3", n, R0, rep))), n, round(Int, RHO_S * n), R0 / MU, Exponential(MU))

# ------------------------------------------------------------------ DDSA with N summed out
"""DDSA log-likelihood with the Binomial final-size term replaced by its sum over the grid (log scale)."""
function ddsa_loglik_N(β, μ, ρ, d, grid)
    p = d.Δt / μ
    (0 < p < 1 && 0 < ρ < 1) || return -Inf
    T_ = promote_type(typeof(β), typeof(μ), typeof(ρ))
    logS0 = log1p(-ρ); logS = logS0; I = T_(ρ); cumλ = zero(T_); acc = zero(T_)
    @inbounds for k in 1:d.nsteps
        λ = β * I * d.Δt; lognew = logS + log(-expm1(-λ))
        d.th[k] > 0 && (acc += d.th[k] * lognew)
        I = I * (1 - p) + exp(lognew); logS -= λ; cumλ += λ
    end
    logτ = logS0 + log(-expm1(-cumλ)); log1mτ = logaddexp(log(ρ), logS)
    ll = d.mc + (acc - d.K * logτ) + marginal_N_loglik(grid, logτ, log1mτ)
    l1p = log1p(-p); lp = log(p)
    for (x, c) in zip(d.du, d.dn); ll += c * (lp + (x - 1) * l1p); end
    for (x, c) in zip(d.cu, d.cn); ll += c * x * l1p; end
    return ll
end
"""τ = S_0 − S_T of the DDSA map (fraction of the whole population newly infected by T)."""
function ddsa_tau(β, μ, ρ, d)
    p = d.Δt / μ; logS0 = log1p(-ρ); logS = logS0; I = ρ; cumλ = 0.0
    for k in 1:d.nsteps
        λ = β * I * d.Δt; lognew = logS + log(-expm1(-λ)); I = I * (1 - p) + exp(lognew); logS -= λ; cumλ += λ
    end
    return exp(logS0 + log(-expm1(-cumλ)))
end
@model function ddsa_N_model(d, grid)
    β ~ Uniform(β_LO, β_HI); μ ~ Uniform(μ_LO, μ_HI); ρ ~ Beta(ρ_A, ρ_B)
    Turing.@addlogprob! ddsa_loglik_N(β, μ, ρ, d, grid)
end

# ------------------------------------------------------------------ continuous-time DSA with n imputed
"""Per-susceptible final size τ_∞ from 1 − τ = exp(−R0 (ρ + (1 − ρ) τ)) (eq. 2.3 in the whole-population
parameterisation), by 50 fixed-point iterations as in the SDS code."""
function tau_inf(β, μ, ρ)
    R0 = β * μ; τ = 1.0
    for _ in 1:50; τ = 1 - exp(-R0 * (ρ + (1 - ρ) * τ)); end
    return τ
end
"""Per-susceptible attack probability by T, 1 − s_T/s_0, from the same RK4 mean field as dsa_loglik."""
function q_T(β, μ, ρ, T; h = 0.02)
    d0 = (n = 0, K = 0, ti = Float64[], T = T, nrec = 0, expo = 0.0)
    return -expm1(dsa_loglik(β, μ, ρ, merge(d0, (n = 1,)); h))     # (n − K) (u − log S0) with n = 1, K = 0: log(s_T/s_0)
end

logC(n, k) = loggamma(n + 1.0) - loggamma(k + 1.0) - loggamma(n - k + 1.0)

function logprior_x(x)          # x = (log β, log μ, logit ρ); priors U(β), U(μ), Beta(1, 10)(ρ) with Jacobians
    β = exp(x[1]); μ = exp(x[2]); ρ = 1 / (1 + exp(-x[3]))
    (β_LO < β < β_HI && μ_LO < μ < μ_HI) || return -Inf
    return x[1] + x[2] + logpdf(Beta(ρ_A, ρ_B), ρ) + log(ρ) + log1p(-ρ)
end

"""One chain of Metropolis-within-Gibbs for (θ, n). variant = :sds (n step of Algorithm 4.1), :sds_window (the same step
with the attack probability by T), :full (complete likelihood) or :geo (no binomial factor).
`lprior` is the log prior density of x = (log β, log μ, logit ρ) (s6_5_sds_own_settings.jl passes the priors of the SDS examples)."""
function dsa_gibbs_chain(rng, de, x0, variant, Nmax; th = 1, lprior = logprior_x)
    K = de.K
    θ(x) = (exp(x[1]), exp(x[2]), 1 / (1 + exp(-x[3])))
    function lp(x, n)
        lpr = lprior(x); isfinite(lpr) || return -Inf
        b, mu, r = θ(x)
        lpr + dsa_loglik(b, mu, r, merge(de, (n = n,))) + (variant === :full ? logC(n, K) : 0.0)
    end
    function draw_n(x)
        b, mu, r = θ(x)
        if variant === :sds
            return K + rand(rng, NegativeBinomial(K, tau_inf(b, mu, r)))
        elseif variant === :sds_window                     # restricted to n ≤ Nmax, the support of ddsa_N and dsa_full
            q = q_T(b, mu, r, de.T)
            for _ in 1:1000
                n = K + rand(rng, NegativeBinomial(K, q)); n <= Nmax && return n
            end
            return Nmax
        elseif variant === :geo                            # n − K ~ Geometric(q_T): the target without C(n, K)
            q = q_T(b, mu, r, de.T)
            for _ in 1:1000
                n = K + rand(rng, NegativeBinomial(1, q)); n <= Nmax && return n
            end
            return Nmax
        else
            q = q_T(b, mu, r, de.T)
            for _ in 1:1000
                n = K + rand(rng, NegativeBinomial(K + 1, q)); n <= Nmax && return n
            end
            return Nmax
        end
    end
    x = copy(x0); n = draw_n(x); lcur = lp(x, n)
    Σ = Matrix(Diagonal([0.05, 0.05, 0.2] .^ 2)); sc = 2.38^2 / 3
    nburn = th * N_BURN; nkeep = th * N_KEEP
    hist = zeros(nburn + nkeep, 3); ns = zeros(Int, nburn + nkeep); acc = 0
    for it in 1:(nburn + nkeep)
        L = cholesky(Symmetric(sc * Σ + 1e-10I)).L
        xp = x .+ L * randn(rng, 3); lpp = lp(xp, n)
        if log(rand(rng)) < lpp - lcur; x = xp; lcur = lpp; it > nburn && (acc += 1); end
        n = draw_n(x); lcur = lp(x, n)
        hist[it, :] = x; ns[it] = n
        if it <= nburn && it >= 500 && it % 100 == 0
            Σ = cov(hist[max(1, it - 2000):it, :]) + Diagonal(fill(1e-8, 3))
        end
    end
    keep = (nburn + th):th:(nburn + nkeep)                  # N_KEEP draws per chain in every variant
    return hist[keep, :], ns[keep], acc / nkeep
end

function fit_dsa_gibbs(rng, de, dd, variant, cut, Nmax; th = nothing, lprior = logprior_x)
    xs = [[log(p.β), log(p.μ), log(p.ρ / (1 - p.ρ))] for p in init_values(rng, dd, dd.meanδ, cut; n_chains = N_CH)]
    th = th !== nothing ? th : variant === :sds_window ? (cut === :peak ? WINDOW_MULT_PEAK : WINDOW_MULT) : 1
    t = @elapsed out = [dsa_gibbs_chain(Xoshiro(rand(rng, UInt)), de, xs[c], variant, Nmax; th, lprior) for c in 1:N_CH]
    arr = cat([hcat(exp.(o[1][:, 1]), exp.(o[1][:, 2]), 1 ./ (1 .+ exp.(-o[1][:, 3])), float.(o[2])) for o in out]...; dims = 3)
    arr = permutedims(arr, (1, 3, 2))                       # draws × chains × (β, μ, ρ, n)
    rh = maximum(MCMCDiagnosticTools.rhat(arr[:, :, 1:3])); eb = minimum(MCMCDiagnosticTools.ess(arr[:, :, 1:3]))
    et = minimum(MCMCDiagnosticTools.ess(arr[:, :, 1:3]; kind = :tail))
    β = vec(arr[:, :, 1]); μ = vec(arr[:, :, 2]); ρ = vec(arr[:, :, 3]); n = vec(arr[:, :, 4])
    Ntot = n .+ de.m
    return (time = t, β = summ(β), μ = summ(μ), R0 = summ(β .* μ), ρ = summ(ρ), N = summ(Ntot),
            rhat = rh, ess = eb, ess_tail = et, div = 0, lpgap = mean(o[3] for o in out))   # lpgap column: acceptance rate
end

"""Continuous-time DSA log-likelihood with n (susceptibles) summed out of the complete Binomial final-size term."""
function dsa_loglik_N(β, μ, ρ, d, grid)
    r = dsa_parts(β, μ, ρ, d); isfinite(r.ll) || return r.ll
    logq = log(-expm1(r.logsT))
    return r.ll + marginal_N_loglik(grid, logq, r.logsT) - d.K * logq
end
@model function dsa_N_model(d, grid)
    β ~ Uniform(β_LO, β_HI); μ ~ Uniform(μ_LO, μ_HI); ρ ~ Beta(ρ_A, ρ_B)
    Turing.@addlogprob! dsa_loglik_N(β, μ, ρ, d, grid)
end

function fit_dsa_N(rng, de, dd, cut, Nmax; grid = NGrid(de.K, de.K, Nmax - de.m))   # n susceptibles ≥ K; N = n + m on the same support as ddsa_N
    m = dsa_N_model(de, grid)
    inits = inits_for(rng, dd, dd.meanδ, cut)
    t = @elapsed ch = sample(rng, m, NUTS(1000, 0.8; max_depth = 12, Δ_max = 1000.0), MCMCSerial(), 1000, 4;
                             progress = false, chain_type = MCMCChains.Chains, initial_params = inits)
    β = vec(ch[:β]); μ = vec(ch[:μ]); ρ = vec(ch[:ρ])
    N = [sample_N(rng, grid, -expm1(dsa_parts(β[i], μ[i], ρ[i], de).logsT)) + de.m for i in eachindex(β)]
    s = summarystats(ch[[:β, :μ, :ρ]])
    arr = permutedims(cat([Array(ch[:, [:β, :μ, :ρ], c]) for c in 1:4]...; dims = 3), (1, 3, 2))
    return (time = t, β = summ(β), μ = summ(μ), R0 = summ(β .* μ), ρ = summ(ρ), N = summ(float.(N)),
            rhat = maximum(s[:, :rhat]), ess = minimum(s[:, :ess_bulk]),
            ess_tail = minimum(MCMCDiagnosticTools.ess(arr; kind = :tail)), div = Int(sum(vec(ch[:numerical_error]))), lpgap = 0.0)
end

function fit_ddsa_N(rng, dd, cut, Nmax)
    grid = NGrid(dd.K, dd.K + 1, Nmax)
    m = ddsa_N_model(dd, grid)
    inits = inits_for(rng, dd, dd.meanδ, cut)
    t = @elapsed ch = sample(rng, m, NUTS(1000, 0.8; max_depth = 12, Δ_max = 1000.0), MCMCSerial(), 1000, 4;
                             progress = false, chain_type = MCMCChains.Chains, initial_params = inits)
    β = vec(ch[:β]); μ = vec(ch[:μ]); ρ = vec(ch[:ρ])
    N = [sample_N(rng, grid, ddsa_tau(β[i], μ[i], ρ[i], dd)) for i in eachindex(β)]
    s = summarystats(ch[[:β, :μ, :ρ]])
    arr = permutedims(cat([Array(ch[:, [:β, :μ, :ρ], c]) for c in 1:4]...; dims = 3), (1, 3, 2))
    return (time = t, β = summ(β), μ = summ(μ), R0 = summ(β .* μ), ρ = summ(ρ), N = summ(float.(N)),
            rhat = maximum(s[:, :rhat]), ess = minimum(s[:, :ess_bulk]),
            ess_tail = minimum(MCMCDiagnosticTools.ess(arr; kind = :tail)), div = Int(sum(vec(ch[:numerical_error]))), lpgap = 0.0)
end

# ------------------------------------------------------------------ runner
if abspath(PROGRAM_FILE) == @__FILE__
    RES = joinpath(OUTDIR, "$RUN.csv"); PARS = (:β, :μ, :R0, :ρ, :N)
    HDR = join(vcat(["n", "R0", "cut", "rep", "model", "K", "T", "time_s"],
                    [string(p, "_", s) for p in PARS for s in ("true", "mean", "sd", "med", "lo", "hi")],
                    ["rhat", "ess", "ess_tail", "div", "accept"]), ",")
    t0 = time()
    isfile(RES) || open(io -> println(io, HDR), RES, "w")
    done = Set{Tuple}()
    for l in Iterators.drop(eachline(RES), 1)
        f = split(l, ","); push!(done, (parse(Int, f[1]), parse(Float64, f[2]), Symbol(f[3]), parse(Int, f[4]), Symbol(f[5])))
    end
    jobs = [(n, R0, cut, rep, mdl) for rep in 1:REPS for n in NS for R0 in R0S for cut in (:peak, :end)
            for mdl in (:ddsa_N, :dsa_sds, :dsa_full, :dsa_window) if !((n, R0, cut, rep, mdl) in done)]
    sort!(jobs, by = j -> (j[4], -j[1]))
    println("pending jobs: $(length(jobs)) (done: $(length(done)))"); flush(stdout)
    lk = ReentrantLock(); next = Threads.Atomic{Int}(1); nfit = Threads.Atomic{Int}(0)
    Threads.@threads for _ in 1:Threads.nthreads()
        while true
            time() - t0 > BUDGET && break
            j = Threads.atomic_add!(next, 1); j > length(jobs) && break
            n, R0, cut, rep, mdl = jobs[j]
            sim = simulate(n, R0, rep); co = cutoffs(n, R0); T = cut === :peak ? co.peak : co.stop
            rng = Xoshiro(hash(("s4fit", n, R0, cut, rep, mdl)))
            dd = obs_steps(sim, T); de = obs_exact(sim, T); Nmax = NMAX_FACTOR * n
            res = try
                mdl === :ddsa_N ? fit_ddsa_N(rng, dd, cut, Nmax) :
                mdl === :dsa_full ? fit_dsa_N(rng, de, dd, cut, Nmax) :
                mdl === :dsa_window ? fit_dsa_gibbs(rng, de, dd, :sds_window, cut, Nmax) : fit_dsa_gibbs(rng, de, dd, :sds, cut, Nmax)
            catch e
                @warn "fit failed" n R0 cut rep mdl e; nothing
            end
            res === nothing && continue
            m = sim.m; tru = (β = R0 / MU, μ = MU, R0 = R0, ρ = m / (n + m), N = float(n + m))
            row = join(Any[n, R0, cut, rep, mdl, dd.K, T, round(res.time, digits = 1),
                           vcat([[tru[p], getfield(res, p)...] for p in PARS]...)...,
                           round(res.rhat, digits = 4), round(res.ess, digits = 0), round(res.ess_tail, digits = 0), res.div,
                           round(res.lpgap, digits = 3)], ",")
            lock(lk) do; open(io -> println(io, row), RES, "a"); end
            Threads.atomic_add!(nfit, 1)
        end
    end
    println("fits completed this run: $(nfit[]) in $(round(time() - t0, digits=0)) s; remaining: $(length(jobs) - nfit[])")
end
