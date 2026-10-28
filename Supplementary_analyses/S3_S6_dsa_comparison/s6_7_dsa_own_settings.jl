# S6: our re-implementation of continuous-time DSA with a Gamma infectious period (s5_nonexp.jl) on the outbreaks of
# s6_6_dsa_own_settings.py, which fits the authors' DSA Python code in the setting of their own synthetic study
# (N = 10,000, I_0 = 50, Gamma IP mean 9 d and variance 6, T_f = 80 d, infection times only).
#   1. Pointwise check of l_I, their likelihood of infection times, sum log(s(t_i) lambda(t_i)) − K log(1 − s(t_K)).
#      llI_renewal recomputes it with the mean-field renewal scheme of dsa_gamma_loglik (h = 0.1 d), with the initial
#      infectives either at age 0 (our convention and the generator's) or with ages uniform on 0–5 d (their PDE's
#      default, which their likelihood uses); with age 0 it equals the infection part of dsa_gamma_loglik (checked).
#   2. Their estimator in our code: the maximiser of llI_renewal (ages uniform on 0–5 d), by Newton steps from their
#      particle-swarm optimum, within their bounds.
#   3. Our analysis of these data: dsa_gamma_model with infection times and the never-infected count (scenario B of S5,
#      n = 9,950 and m = 50 known; S5 priors; NUTS, 4 chains of 1,000 + 1,000; starts at R0 from the final size,
#      mean IP 10 d, shape 10, rho = m/N, jittered).
# Parameters are reported on our scale: beta = beta_PDE (1 + rho_PDE), rho = rho_PDE/(1 + rho_PDE), R0 = beta × mean IP.
# Usage (from the supplementary folder, after s6_6): julia --project=@. -t 5 S3_S6_dsa_comparison/s6_7_dsa_own_settings.jl
include(joinpath(@__DIR__, "s5_nonexp.jl"))
using CSV, DataFrames
const DATA = joinpath(@__DIR__, "external", "data"); const OUT = joinpath(@__DIR__, "out")
const SEEDS = 1:parse(Int, get(ENV, "SEEDS", "5"))
const LB = [5e-4, 0.01, 2.0, 1.0]; const UB = [1e-2, 2.0, 20.0, 15.0]   # their bounds on (rho, beta, mean, variance)

jsonrows(s) = [Dict(m[1] => parse(Float64, m[2]) for m in eachmatch(r"\"(\w+)\": ([-0-9.eE+]+)", b))
               for b in eachmatch(r"\{[^}]*\}", s) .|> (x -> x.match)]
ours(x) = (β = x[2] * (1 + x[1]), ρ = x[1] / (1 + x[1]), μ = x[3], k = x[3]^2 / x[4])   # their (ρ, β, mean, var) → ours

"""l_I of Di Lauro et al. with the renewal scheme of dsa_gamma_loglik; `seed` = :age0 or :unif5 (initial ages on 0–5 d)."""
function llI_renewal(β, μ, k, ρ, ti, T; h = 0.1, seed = :age0)
    T_ = promote_type(typeof(β), typeof(μ), typeof(k), typeof(ρ))
    θ = μ / k; nt = ceil(Int, T / h) + 1
    μv = ForwardDiff.value(μ); kv = ForwardDiff.value(k)
    L = min(nt + 60, ceil(Int, (μv + 12 * μv / sqrt(kv) + 5) / h))
    lpdf(a) = (k - 1) * log(a) - a / θ - loggamma(k) - k * log(θ)
    g = [l == 0 ? (k == 1 ? 1 / θ : zero(T_)) : exp(lpdf(l * h)) for l in 0:(L + 1)]
    G = Vector{T_}(undef, L + 2); G[1] = one(T_)
    for l in 1:(L + 1); G[l+1] = max(G[l] - h * (g[l] + g[l+1]) / 2, zero(T_)); end
    Gc = [(G[l+1] + G[l+2]) / 2 for l in 0:L]
    Gat(x) = (y = x / h; i = floor(Int, y) + 1; i >= L + 2 ? zero(T_) : (w = y - (i - 1); (1 - w) * G[i] + w * G[i+1]))
    as = [(j - 0.5) * 0.1 for j in 1:50]
    seedS(n) = seed === :age0 ? (n - 1 <= L + 1 ? G[n] : zero(T_)) : sum(Gat(a + (n - 1) * h) / Gat(a) for a in as) / length(as)
    logS0 = log1p(-ρ); logS = Vector{T_}(undef, nt); Iv = Vector{T_}(undef, nt); dC = Vector{T_}(undef, nt)
    logS[1] = logS0; Iv[1] = ρ; dC[1] = zero(T_)
    for n in 2:nt
        base = ρ * seedS(n)
        for j in max(2, n - L):(n - 1); base += dC[j] * Gc[n - j + 1]; end
        In = Iv[n-1]
        for _ in 1:3
            ls = logS[n-1] - β * h * (Iv[n-1] + In) / 2
            In = base + (exp(logS[n-1]) - exp(ls)) * Gc[1]
        end
        Iv[n] = In; logS[n] = logS[n-1] - β * h * (Iv[n-1] + In) / 2
        dC[n] = exp(logS[n-1]) - exp(logS[n])
    end
    at(t) = (x = t / h; i = min(floor(Int, x) + 1, nt - 1); w = x - (i - 1); (i, w))
    ll = zero(T_); lβ = log(β)
    for t in ti
        i, w = at(t); ll += lβ + (1 - w) * logS[i] + w * logS[i+1] + log((1 - w) * Iv[i] + w * Iv[i+1]) - logS0
    end
    i, w = at(ti[end]); logsK = (1 - w) * logS[i] + w * logS[i+1] - logS0
    return ll - length(ti) * log(-expm1(logsK))
end
llI_x(x, ti; kw...) = (p = ours(x); llI_renewal(p.β, p.μ, p.k, p.ρ, ti, 80.0; kw...))

loadti(s) = sort(vec(readdlm_floats(joinpath(DATA, "dsaown_seed$(s)_infection_times.txt"))))
readdlm_floats(f) = parse.(Float64, filter(!isempty, strip.(readlines(f))))

# ------------------------------------------------------------------ 1. pointwise check
pw = DataFrame()
for s in SEEDS, grid in (800, 10000)
    f = joinpath(OUT, "s6_6_dsa_own_ll_seed$(s)_grid$(grid).json"); isfile(f) || continue
    ti = loadti(s)
    for r in jsonrows(read(f, String))
        x = [r["rho"], r["beta"], r["mean"], r["variance"]]
        push!(pw, (seed = s, grid = grid, rho = x[1], beta = x[2], mean = x[3], variance = x[4], llI_python = r["llI_published"],
                   llI_ours_unif5 = llI_x(x, ti; seed = :unif5), llI_ours_age0 = llI_x(x, ti; seed = :age0)))
    end
end
let ti = loadti(1), p = ours([0.005, 0.25, 9.0, 6.0])         # copy check against dsa_gamma_loglik (age 0)
    d = (n = length(ti), K = length(ti), T = ti[end], ti = ti)
    a = dsa_gamma_loglik(p.β, p.μ, p.k, p.ρ, d, :B)
    b = llI_renewal(p.β, p.μ, p.k, p.ρ, ti, 80.0) + length(ti) * log(-expm1(dsa_gamma_loglik(p.β, p.μ, p.k, p.ρ, merge(d, (n = d.K + 1,)), :B) - a))
    @printf("copy check: dsa_gamma_loglik infection part − llI_renewal(age 0) = %.2e\n", a - b)
end
CSV.write(joinpath(OUT, "s6_7_dsa_own_ll.csv"), pw)
for g in groupby(pw, [:seed, :grid])
    d1 = g.llI_ours_unif5 .- g.llI_python; d0 = g.llI_ours_age0 .- g.llI_python
    @printf("seed %d grid %5d: ours(unif 0–5) − Python %.3f to %.3f (spread %.3f); ours(age 0) − Python %.3f to %.3f\n",
            g.seed[1], g.grid[1], minimum(d1), maximum(d1), maximum(d1) - minimum(d1), minimum(d0), maximum(d0))
end

# ------------------------------------------------------------------ 2. their estimator in our code
fits = isfile(joinpath(OUT, "s6_6_dsa_own_fits.jsonl")) ? jsonrows(read(joinpath(OUT, "s6_6_dsa_own_fits.jsonl"), String)) : []
est = DataFrame()
σ(z) = 1 / (1 + exp(-z))
for r in fits
    s = Int(r["seed"]); ti = loadti(s)
    xp = [r["rho"], r["beta"], r["mean"], r["variance"]]
    u = clamp.((xp .- LB) ./ (UB .- LB), 1e-6, 1 - 1e-6); z = log.(u ./ (1 .- u))
    F(z) = llI_x(LB .+ (UB .- LB) .* σ.(z), ti; seed = :unif5)
    fz = F(z)
    for it in 1:60
        gr = ForwardDiff.gradient(F, z); H = ForwardDiff.hessian(F, z)
        Hn = Symmetric(-H); λmin = eigmin(Hn)
        step = (Hn + max(0.0, 1e-6 - λmin) * I) \ gr
        t = 1.0; zn = z .+ step; fn = F(zn)
        while !(fn > fz - 1e-10) && t > 1e-8; t /= 2; zn = z .+ t .* step; fn = F(zn); end
        fn > fz - 1e-10 || break
        done = abs(fn - fz) < 1e-9; z, fz = zn, fn; done && break
    end
    xo = LB .+ (UB .- LB) .* σ.(z)
    po = ours(xo); pp = ours(xp)
    push!(est, (seed = s, K = length(ti), python_R0 = pp.β * pp.μ, python_mean = xp[3], python_var = xp[4], python_rho = pp.ρ,
                python_nll = r["nll"], ours_R0 = po.β * po.μ, ours_mean = xo[3], ours_var = xo[4], ours_rho = po.ρ,
                ours_llI_at_python = llI_x(xp, ti; seed = :unif5), ours_llI_at_ours = fz,
                at_bound = any(min.(xo .- LB, UB .- xo) ./ (UB .- LB) .< 1e-3)))
end
CSV.write(joinpath(OUT, "s6_7_dsa_own_mle.csv"), est)
nrow(est) > 0 && show(stdout, MIME("text/plain"), est; allcols = true); println()

# ------------------------------------------------------------------ 3. our analysis (infection times + never-infected count)
if get(ENV, "FITS", "1") == "1"
    res = Vector{Any}(undef, length(SEEDS))
    Threads.@threads for j in eachindex(SEEDS)
        s = SEEDS[j]; ti = loadti(s); rng = Xoshiro(hash(("s6_7", s)))
        d = (n = 9950, m = 50, T = 80.0, ti = ti, K = length(ti))
        z = d.K / d.n; R0h = clamp(-log(1 - z) / z, 1.05, 6.0)
        inits = [InitFromParams((β = R0h / 10 * exp(0.1randn(rng)), μ = 10 * exp(0.1randn(rng)), k = 10 * exp(0.1randn(rng)),
                                 ρ = 0.005 * exp(0.1randn(rng)))) for _ in 1:4]
        t = @elapsed ch = sample(rng, dsa_gamma_model(d, :B), NUTS(1000, 0.8; max_depth = 12, Δ_max = 1000.0), MCMCSerial(), 1000, 4;
                                 progress = false, chain_type = MCMCChains.Chains, initial_params = inits)
        β = vec(ch[:β]); μ = vec(ch[:μ]); kk = vec(ch[:k]); st = summarystats(ch[[:β, :μ, :k, :ρ]])
        q(v) = (median(v), quantile(v, 0.025), quantile(v, 0.975))
        res[j] = (seed = s, K = d.K, R0 = q(β .* μ), mean = q(μ), var = q(μ .^ 2 ./ kk), rhat = maximum(st[:, :rhat]),
                  div = Int(sum(vec(ch[:numerical_error]))), time = t)
        @printf("seed %d (ours, infection times + never infected): R0 %.3f [%.3f, %.3f], mean IP %.2f [%.2f, %.2f], var %.2f, R̂ %.3f, %d div, %.0f s\n",
                s, res[j].R0..., res[j].mean..., res[j].var[1], res[j].rhat, res[j].div, t)
    end
    CSV.write(joinpath(OUT, "s6_7_dsa_own_bayes.csv"),
              DataFrame([(seed = r.seed, K = r.K, R0_med = r.R0[1], R0_lo = r.R0[2], R0_hi = r.R0[3], mean_med = r.mean[1],
                          mean_lo = r.mean[2], mean_hi = r.mean[3], var_med = r.var[1], rhat = r.rhat, div = r.div, time_s = r.time)
                         for r in res]))
end
