# Supplement S5: non-exponential infectious periods, at the synthetic setting of Di Lauro et al. (2022, §4.1):
# N = 10,000 (n = 9,950 susceptibles, m = 50 initial infectives), β = 0.25 per day, Gamma infectious period with mean
# 9 d and variance 6 (shape 13.5), outbreaks observed to extinction. Three data scenarios, as in Section 3(a):
#   A  full line list (infection times, final size, recovery intervals)
#   B  infection times and final size only
#   C  recovery intervals only
# Models (priors: β ~ U(1e-4, 2.5), mean IP μ ~ U(1, 30) d, ρ ~ Beta(1, 10); shape below):
#   ddsa  GLCT map with a NegBin DPH infectious period (r stages, p = r/μ per day), r summed out over 1, …, r_max with
#         r_max = min(10, min δ) when intervals are observed and 10 otherwise (the main-text model)
#   ddsa_box  DDSA with the infection-age (Boxcar) map and a discretised Gamma infectious period (shape k ~ U(1, 30)),
#         I_k = ρ P(δ_seed ≥ k) + Σ_a new_{k−a} P(δ ≥ a). The interval law follows the recording convention: with the
#         infection time uniform within its day, P(δ = d) = G(d+1) − 2G(d) + G(d−1), G(x) = ∫_0^x F_Gamma, and same-day
#         recoveries recorded as 1 d (P(δ = 1) = G(2) − G(1)); seeds, infected at t = 0, have δ = ⌊W⌋ + 1. Its support
#         starts at 1 d whatever the shape, so it has no r ≤ min δ cap: the arm tests whether the cap belongs to the
#         GLCT structure (G; at most one stage transition per step) rather than to the likelihood (L) or the map (M).
#   dsa   our implementation of continuous-time DSA with a Gamma infectious period (shape k ~ U(1, 30)): the mean-field renewal equation
#         I(t) = ρ G(t) + ∫ j(u) G(t − u) du, j = β S I, S(t) = S_0 exp(−β ∫ I), solved on a grid of h days (the method
#         of characteristics for the age-structured PDE of Di Lauro et al.), log S and I interpolated at the exact
#         infection times; Gamma PDF for the intervals.
#   REPS=20  BUDGET=420  RUN=s5_main  H=0.1
# Usage (from the supplementary folder): REPS=3 RUN=s5_pilot julia --project=@. -t 10 S3_S6_dsa_comparison/s5_nonexp.jl
include(joinpath(@__DIR__, "s3_core.jl"))
using SpecialFunctions: loggamma
using ForwardDiff
const NS5, MS5, β5, μ5, V5 = 9950, 50, 0.25, 9.0, 6.0
const K5 = μ5^2 / V5                          # Gamma shape 13.5
const R_MAX5 = 10
const k_LO, k_HI = 1.0, 30.0

simulate5(rep) = sim_outbreak(Xoshiro(hash(("s5", rep))), NS5, MS5, β5, Gamma(K5, μ5 / K5))

# ------------------------------------------------------------------ DDSA, NegBin DPH, r summed out
"""DDSA log-likelihood with an r-stage NegBin DPH infectious period (per-step progression p = rΔt/μ), log scale.
Terms switched by `use`: :A (all), :B (infection times + final size), :C (intervals only)."""
function ddsa_dph_loglik(β, μ, ρ, r, d, use)
    p = r * d.Δt / μ
    (0 < p < 1 && 0 < ρ < 1) || return -Inf
    T_ = promote_type(typeof(β), typeof(μ), typeof(ρ))
    ll = zero(T_)
    if use !== :C
        logS0 = log1p(-ρ); logS = logS0; cumλ = zero(T_); acc = zero(T_)
        x = zeros(T_, r); x[1] = ρ
        @inbounds for k in 1:d.nsteps
            λ = β * sum(x) * d.Δt; lognew = logS + log(-expm1(-λ))
            d.th[k] > 0 && (acc += d.th[k] * lognew)
            for j in r:-1:2; x[j] = x[j] * (1 - p) + x[j-1] * p; end
            x[1] = x[1] * (1 - p) + exp(lognew)
            logS -= λ; cumλ += λ
        end
        logτ = logS0 + log(-expm1(-cumλ)); log1mτ = logaddexp(log(ρ), logS)
        ll += d.mc + (acc - d.K * logτ) + d.logCK + d.K * logτ + (d.Ntot - d.K) * log1mτ
    end
    if use !== :B                                  # NegBin total trials: C(w−1, r−1) p^r (1−p)^(w−r), w ≥ r
        lp = log(p); l1p = log1p(-p)
        for (w, c) in zip(d.du, d.dn)
            w < r && return -Inf
            ll += c * (loggamma(w + 0.0) - loggamma(r + 0.0) - loggamma(w - r + 1.0) + r * lp + (w - r) * l1p)
        end
        for (w, c) in zip(d.cu, d.cn)          # survivor P(W > w) = P(fewer than r successes in w trials)
            w < r && continue
            ll += c * log(sum(exp(loggamma(w + 1.0) - loggamma(j + 1.0) - loggamma(w - j + 1.0) + j * lp + (w - j) * l1p) for j in 0:(r - 1)))
        end
    end
    return ll
end
@model function ddsa_dph_model(d, use, rmax)
    β ~ Uniform(β_LO, β_HI); μ ~ Uniform(μ_LO, μ_HI); ρ ~ Beta(ρ_A, ρ_B)
    logmarg, lps = marginal_r_loglik(r -> ddsa_dph_loglik(β, μ, ρ, r, d, use), 1:rmax)
    Turing.@addlogprob! logmarg
    return lps
end

# ------------------------------------------------------------------ DDSA, Boxcar map, discretised Gamma
"""Daily interval laws of a Gamma(k, μ/k) infectious period under the recording convention (see header): PMF of δ for
cases (pc) and for seeds (ps), d = 1..A. F is the trapezoid-integrated PDF on a grid of h days (differentiable in k)."""
function discretised_gamma(μ, k, A; h = 0.05)
    T_ = promote_type(typeof(μ), typeof(k)); θ = μ / k
    n = round(Int, (A + 1) / h)
    pdfv(a) = a == 0 ? (k == 1 ? 1 / θ : zero(T_)) : exp((k - 1) * log(a) - a / θ - loggamma(k) - k * log(θ))
    F = zeros(T_, n + 1); G = zeros(T_, n + 1); prev = pdfv(0.0)
    for i in 1:n
        cur = pdfv(i * h); F[i+1] = min(F[i] + h * (prev + cur) / 2, one(T_)); prev = cur
        G[i+1] = G[i] + h * (F[i] + F[i+1]) / 2
    end
    step = round(Int, 1 / h); Fi(d) = F[d * step + 1]; Gi(d) = d <= 0 ? zero(T_) : G[d * step + 1]
    pc = [d == 1 ? Gi(2) - Gi(1) : Gi(d + 1) - 2Gi(d) + Gi(d - 1) for d in 1:A]
    ps = [Fi(d) - Fi(d - 1) for d in 1:A]
    return max.(pc, zero(T_)), max.(ps, zero(T_))
end

"""DDSA log-likelihood with the infection-age (Boxcar) map and the discretised Gamma interval law (log scale for S).
d: obs_steps data with the seeds' intervals separated (fields du, dn for cases; su, sn for seeds)."""
function ddsa_box_loglik(β, μ, k, ρ, d, use)
    (0 < ρ < 1 && μ > 0 && k >= 1) || return -Inf
    T_ = promote_type(typeof(β), typeof(μ), typeof(k), typeof(ρ))
    A = max(d.nsteps, maximum(vcat(d.du, d.su))) + 1
    pc, ps = discretised_gamma(μ, k, A)
    ll = zero(T_)
    if use !== :C
        Sc = reverse(cumsum(reverse(pc)))                      # P(δ ≥ a), a = 1..A, as tail sums (nonnegative)
        Ss = reverse(cumsum(reverse(ps)))
        logS0 = log1p(-ρ); logS = logS0; cumλ = zero(T_); acc = zero(T_); new = zeros(T_, d.nsteps)
        @inbounds for t in 1:d.nsteps
            I = ρ * Ss[t]
            for a in 1:(t - 1); I += new[t - a] * Sc[a]; end
            λ = β * I * d.Δt; lognew = logS + log(-expm1(-λ))
            d.th[t] > 0 && (acc += d.th[t] * lognew)
            new[t] = exp(lognew); logS -= λ; cumλ += λ
        end
        logτ = logS0 + log(-expm1(-cumλ)); log1mτ = logaddexp(log(ρ), logS)
        ll += d.mc + (acc - d.K * logτ) + d.logCK + d.K * logτ + (d.Ntot - d.K) * log1mτ
    end
    if use !== :B
        for (x, c) in zip(d.du, d.dn); ll += c * log(pc[x]); end
        for (x, c) in zip(d.su, d.sn); ll += c * log(ps[x]); end
    end
    return ll
end
@model function ddsa_box_model(d, use)
    β ~ Uniform(β_LO, β_HI); μ ~ Uniform(μ_LO, μ_HI); k ~ Uniform(k_LO, k_HI); ρ ~ Beta(ρ_A, ρ_B)
    Turing.@addlogprob! ddsa_box_loglik(β, μ, k, ρ, d, use)
end
"""obs_steps data with the seeds' intervals (⌊W⌋ + 1) separated from the cases' (no censoring: outbreak to extinction)."""
function obs_steps_box(sim, T)
    dd = obs_steps(sim, T)
    sd = [floor(Int, r) + 1 for r in sim.seed_rec if r < T]
    cd = [max(1, floor(Int, r) + 1 - (floor(Int, t) + 1)) for (t, r) in zip(sim.inf_t, sim.rec_t) if t < T && r < T]
    du = sort(unique(cd)); dn = [count(==(x), cd) for x in du]
    su = sort(unique(sd)); sn = [count(==(x), sd) for x in su]
    return merge(dd, (du = du, dn = dn, su = su, sn = sn))
end

# ------------------------------------------------------------------ continuous-time DSA, Gamma infectious period
"""Continuous-time DSA log-likelihood with a Gamma(k, μ/k) infectious period. Mean field on a grid of step h:
S_n = S_{n−1} exp(−β h (I_{n−1} + I_n)/2); the mass infected in cell j, ΔC_j = S_{j−1} − S_j, contributes to I_n with
the Gamma survival averaged over its ages, (G((n−j)h) + G((n−j+1)h))/2; the current cell makes the step implicit and
is solved by three fixed-point iterations (second-order scheme). G is the trapezoid-integrated Gamma PDF, so it is
differentiable in the shape k. log S and I are interpolated linearly at the exact infection times; Gamma PDF for
the complete recovery intervals."""
function dsa_gamma_loglik(β, μ, k, ρ, d, use; h = 0.1)
    (0 < ρ < 1 && μ > 0 && k >= 1) || return -Inf
    T_ = promote_type(typeof(β), typeof(μ), typeof(k), typeof(ρ))
    θ = μ / k; ll = zero(T_)
    if use !== :C
        nt = ceil(Int, d.T / h) + 1
        μv = ForwardDiff.value(μ); kv = ForwardDiff.value(k)
        L = min(nt, ceil(Int, (μv + 12 * μv / sqrt(kv) + 5) / h))
        lpdf(a) = (k - 1) * log(a) - a / θ - loggamma(k) - k * log(θ)
        g = [l == 0 ? (k == 1 ? 1 / θ : zero(T_)) : exp(lpdf(l * h)) for l in 0:(L + 1)]
        G = Vector{T_}(undef, L + 2); G[1] = one(T_)
        for l in 1:(L + 1); G[l+1] = max(G[l] - h * (g[l] + g[l+1]) / 2, zero(T_)); end
        Gc = [(G[l+1] + G[l+2]) / 2 for l in 0:L]          # Gc[l+1]: mean survival of a cell whose ages span (lh, (l+1)h)
        logS0 = log1p(-ρ); logS = Vector{T_}(undef, nt); Iv = Vector{T_}(undef, nt); dC = Vector{T_}(undef, nt)
        logS[1] = logS0; Iv[1] = ρ; dC[1] = zero(T_)
        for n in 2:nt
            base = ρ * (n - 1 <= L + 1 ? G[n] : zero(T_))
            for j in max(2, n - L):(n - 1)                  # cell j spans (t_{j−1}, t_j); at t_n its ages span ((n−j)h, (n−j+1)h)
                base += dC[j] * Gc[n - j + 1]
            end
            In = Iv[n-1]
            for _ in 1:3
                ls = logS[n-1] - β * h * (Iv[n-1] + In) / 2
                In = base + (exp(logS[n-1]) - exp(ls)) * Gc[1]
            end
            Iv[n] = In; logS[n] = logS[n-1] - β * h * (Iv[n-1] + In) / 2
            dC[n] = exp(logS[n-1]) - exp(logS[n])
        end
        lβ = log(β)
        for t in d.ti
            x = t / h; i = min(floor(Int, x) + 1, nt - 1); w = x - (i - 1)
            ll += lβ + (1 - w) * logS[i] + w * logS[i+1] + log((1 - w) * Iv[i] + w * Iv[i+1]) - logS0
        end
        x = d.T / h; i = min(floor(Int, x) + 1, nt - 1); w = x - (i - 1)
        ll += (d.n - d.K) * ((1 - w) * logS[i] + w * logS[i+1] - logS0)
    end
    if use !== :B
        lθ = log(θ); lgk = loggamma(k)
        for wv in d.w; ll += (k - 1) * log(wv) - wv / θ - lgk - k * lθ; end
    end
    return ll
end
@model function dsa_gamma_model(d, use)
    β ~ Uniform(β_LO, β_HI); μ ~ Uniform(μ_LO, μ_HI); k ~ Uniform(k_LO, k_HI); ρ ~ Beta(ρ_A, ρ_B)
    Turing.@addlogprob! dsa_gamma_loglik(β, μ, k, ρ, d, use)
end

"""Exact data to extinction, with each complete interval kept (seeds included)."""
function obs_exact_w(sim, T)
    de = obs_exact(sim, T)
    w = vcat([r - t for (t, r) in zip(sim.inf_t, sim.rec_t) if r < T], [r for r in sim.seed_rec if r < T])
    return merge(de, (w = w,))
end

# ------------------------------------------------------------------ fits
function fit5(rng, mdl, use, sim, T)
    dd = obs_steps(sim, T)
    rmax = use === :B ? R_MAX5 : min(R_MAX5, dd.du[1])
    vals = init_values(rng, dd, dd.meanδ, :end)
    if mdl === :ddsa
        m = ddsa_dph_model(dd, use, rmax); inits = InitFromParams.(vals)
    elseif mdl === :ddsa_box
        db = obs_steps_box(sim, T); m = ddsa_box_model(db, use)
        k0 = clamp(dd.meanδ^2 / max(var(vcat(fill.(db.du, db.dn)...)) - 1 / 6, 0.2), 1.0, 25.0)   # minus rounding variance
        inits = [InitFromParams(merge(v, (k = clamp(k0 * exp(0.1 * randn(rng)), 1.05, 29.0),))) for v in vals]
    else
        de = obs_exact_w(sim, T); m = dsa_gamma_model(de, use)
        k0 = clamp(mean(de.w)^2 / var(de.w), 1.0, 25.0)
        inits = [InitFromParams(merge(v, (k = clamp(k0 * exp(0.1 * randn(rng)), 0.6, 29.0),))) for v in vals]
    end
    t = @elapsed ch = sample(rng, m, NUTS(1000, 0.8; max_depth = 12, Δ_max = 1000.0), MCMCSerial(), 1000, 4;
                             progress = false, chain_type = MCMCChains.Chains, initial_params = inits)
    β = vec(ch[:β]); μ = vec(ch[:μ]); ρ = vec(ch[:ρ])
    if mdl === :ddsa
        lps = returned(m, ch); rd = vec([sample_index(rng, l) for l in lps])
        sdv = [sqrt(r * (1 - r / μi)) / (r / μi) for (r, μi) in zip(rd, μ)]     # NegBin total-trials SD
        pr = [mean(rd .== r) for r in 1:R_MAX5]; shape = float.(rd)
        pars = [:β, :μ, :ρ]
    else                                                               # Gamma shape (continuous DSA and Boxcar arm)
        kk = vec(ch[:k]); sdv = μ ./ sqrt.(kk); pr = fill(NaN, R_MAX5); shape = kk
        pars = [:β, :μ, :k, :ρ]
    end
    s = summarystats(ch[pars])
    arr = permutedims(cat([Array(ch[:, pars, c]) for c in 1:4]...; dims = 3), (1, 3, 2))
    return (time = t, β = summ(β), μ = summ(μ), R0 = summ(β .* μ), ρ = summ(ρ), sd = summ(sdv), shape = summ(shape),
            rmax = rmax, pr = pr, rhat = maximum(s[:, :rhat]), ess = minimum(s[:, :ess_bulk]),
            ess_tail = minimum(MCMCDiagnosticTools.ess(arr; kind = :tail)), div = Int(sum(vec(ch[:numerical_error]))))
end

if abspath(PROGRAM_FILE) == @__FILE__
    REPS = parse(Int, get(ENV, "REPS", "20")); BUDGET = parse(Float64, get(ENV, "BUDGET", "420"))
    RUN = get(ENV, "RUN", "s5_main"); RES = joinpath(OUTDIR, "$RUN.csv"); PARS = (:β, :μ, :R0, :ρ, :sd, :shape)
    HDR = join(vcat(["rep", "scenario", "model", "K", "T", "min_delta", "rmax", "time_s"],
                    [string(p, "_", s) for p in PARS for s in ("true", "mean", "sd", "med", "lo", "hi")],
                    ["P_r$r" for r in 1:R_MAX5], ["rhat", "ess", "ess_tail", "div"]), ",")
    t0 = time()
    isfile(RES) || open(io -> println(io, HDR), RES, "w")
    done = Set{Tuple}()
    for l in Iterators.drop(eachline(RES), 1)
        f = split(l, ","); push!(done, (parse(Int, f[1]), Symbol(f[2]), Symbol(f[3])))
    end
    jobs = [(rep, use, mdl) for rep in 1:REPS for mdl in (:dsa, :ddsa, :ddsa_box) for use in (:A, :B, :C) if !((rep, use, mdl) in done)]
    println("pending jobs: $(length(jobs)) (done: $(length(done)))"); flush(stdout)
    lk = ReentrantLock(); next = Threads.Atomic{Int}(1); nfit = Threads.Atomic{Int}(0)
    Threads.@threads for _ in 1:Threads.nthreads()
        while true
            time() - t0 > BUDGET && break
            j = Threads.atomic_add!(next, 1); j > length(jobs) && break
            rep, use, mdl = jobs[j]
            sim = simulate5(rep); T = ceil(maximum(vcat(sim.rec_t, sim.seed_rec))) + 1.0
            res = try
                fit5(Xoshiro(hash(("s5fit", rep, use, mdl))), mdl, use, sim, T)
            catch e
                @warn "fit failed" rep use mdl e; nothing
            end
            res === nothing && continue
            dd = obs_steps(sim, T)
            tru = (β = β5, μ = μ5, R0 = β5 * μ5, ρ = MS5 / (NS5 + MS5), sd = sqrt(V5), shape = K5)
            row = join(Any[rep, use, mdl, dd.K, T, dd.du[1], res.rmax, round(res.time, digits = 1),
                           vcat([[tru[p], getfield(res, p)...] for p in PARS]...)..., res.pr...,
                           round(res.rhat, digits = 4), round(res.ess, digits = 0), round(res.ess_tail, digits = 0), res.div], ",")
            lock(lk) do; open(io -> println(io, row), RES, "a"); end
            Threads.atomic_add!(nfit, 1)
        end
    end
    println("fits completed this run: $(nfit[]) in $(round(time() - t0, digits=0)) s; remaining: $(length(jobs) - nfit[])")
end
