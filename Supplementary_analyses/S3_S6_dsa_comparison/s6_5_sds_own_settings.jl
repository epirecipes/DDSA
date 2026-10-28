# S6: our re-implementation of continuous-time DSA (s3_core.jl, eq. 3.3 of KhudaBukhsh et al. 2020) and of their
# Algorithm 4.1 (s4_popsize.jl), on the datasets of s6_4_sds_own_settings.R, which fits the authors' SDS R package in
# the settings of its own example scripts (Example2.R: Sellke outbreaks, n = 1,000, beta = 1.1, gamma = 0.8,
# rho = 0.05, T = 30; example1.R: WSU, T = 104). Arms:
#   ours_sdsprior   n known: our likelihood with the priors of the SDS examples, Gamma(0.001, 0.001) on beta_SDS,
#                   gamma and rho_SDS < 1 (their per-susceptible scale), sampled with NUTS (4 chains, 2,000 + 2,000)
#   ours_s3prior    n known: our S3 model as used in Table S9 (U(beta), U(mu), Beta(1, 10) on rho)
#   alg41_sdsprior  n imputed: our Algorithm 4.1 (n − K ~ NegBin(K, tau_inf), no C(n, K)) with the SDS priors
#   alg41_s4prior   n imputed: our Algorithm 4.1 with the S4 priors
# Pointwise check: our log-likelihood at the 20 parameter points per dataset at which s6_4 evaluated theirs, with the
# survival term at T (eq. 3.3, our implementation) and at the last infection time (their code).
# Usage (from the supplementary folder, after s6_4_sds_own_settings.R): julia --project=@. -t 4 S3_S6_dsa_comparison/s6_5_sds_own_settings.jl
include(joinpath(@__DIR__, "s4_popsize.jl"))
using CSV, DataFrames

const DATA = joinpath(@__DIR__, "external", "data")
const GPRIOR = Gamma(0.001, 1000.0)        # Gamma(shape 0.001, rate 0.001)

"""Line list of s6_4 (rows: infection time, recovery time; seeds infected at 0; never-infected rows at T) as a simulator
object for obs_exact/obs_steps. n is the number of non-seed rows unless given."""
function load_set(file; n = nothing)
    df = CSV.read(file, DataFrame); T = df.Tmax[1]
    seed = df.inf .== 0
    o = sortperm(df.inf[.!seed])
    inf_t = df.inf[.!seed][o]; rec_t = df.rec[.!seed][o]
    sim = (n = n === nothing ? length(inf_t) : n, m = count(seed), inf_t = inf_t, rec_t = rec_t, seed_rec = df.rec[seed])
    return sim, float(T)
end

@model function dsa_sdsprior_model(d)
    βS ~ GPRIOR; γ ~ GPRIOR; ρS ~ truncated(GPRIOR, 0, 1)
    Turing.@addlogprob! dsa_loglik(βS * (1 + ρS), 1 / γ, ρS / (1 + ρS), d)
end

"""The SDS priors as a log density of x = (log β, log μ, logit ρ): β_SDS = β/(1 + ρ_SDS), γ = 1/μ, ρ_SDS = ρ/(1 − ρ) < 1;
Jacobian β_SDS γ ρ_SDS."""
function logprior_sds(x)
    x[3] < 0 || return -Inf
    ρS = exp(x[3]); βS = exp(x[1]) / (1 + ρS); γ = exp(-x[2])
    return logpdf(GPRIOR, βS) + logpdf(GPRIOR, γ) + logpdf(GPRIOR, ρS) + log(βS) + log(γ) + log(ρS)
end

function fit_sdsprior(rng, de, dd)
    inits = [(βS = p.β * (1 - p.ρ), γ = 1 / p.μ, ρS = p.ρ / (1 - p.ρ)) for p in init_values(rng, dd, dd.meanδ, :end)]
    t = @elapsed ch = sample(rng, dsa_sdsprior_model(de), NUTS(1000, 0.8; max_depth = 12, Δ_max = 1000.0), MCMCSerial(),
                             2000, 4; progress = false, chain_type = MCMCChains.Chains, initial_params = InitFromParams.(inits))
    βS = vec(ch[:βS]); γ = vec(ch[:γ]); ρS = vec(ch[:ρS]); β = βS .* (1 .+ ρS)
    s = summarystats(ch[[:βS, :γ, :ρS]])
    return (time = t, β = summ(β), μ = summ(1 ./ γ), R0 = summ(β ./ γ), ρ = summ(ρS ./ (1 .+ ρS)), N = fill(NaN, 5),
            rhat = maximum(s[:, :rhat]), ess = minimum(s[:, :ess_bulk]), div = Int(sum(vec(ch[:numerical_error]))))
end

sets = vcat([(name = "sellke_rep$r", file = joinpath(DATA, "sdsown_sellke_rep$r.csv"), known = true, n_init = nothing)
             for r in 1:parse(Int, get(ENV, "REPS", "10"))],
            [(name = "wsu", file = joinpath(DATA, "sdsown_wsu.csv"), known = false, n_init = 7803)])

if abspath(PROGRAM_FILE) == @__FILE__
    # ------------------------------------------------------------------ pointwise check
    llt = CSV.read(joinpath(@__DIR__, "out", "s6_4_sds_own_ll.csv"), DataFrame)
    llt.ll_ours_T = zeros(nrow(llt)); llt.ll_ours_tlast = zeros(nrow(llt))
    for s in sets
        sim, T = load_set(s.file)
        for i in findall(==(s.name), llt.set)
            r = llt[i, :]; de = merge(obs_exact(sim, T), (n = r.n,))
            β = r.beta_sds * (1 + r.rho_sds); μ = 1 / r.gamma; ρ = r.rho_sds / (1 + r.rho_sds)
            p = dsa_parts(β, μ, ρ, de); pl = dsa_parts(β, μ, ρ, merge(de, (T = de.ti[end],)))
            llt.ll_ours_T[i] = p.ll + (de.n - de.K) * p.logsT
            llt.ll_ours_tlast[i] = p.ll + (de.n - de.K) * pl.logsT
        end
    end
    CSV.write(joinpath(@__DIR__, "out", "s6_5_sds_own_ll.csv"), llt)
    for g in groupby(llt, :set)
        a = g.ll_ours_tlast .- g.ll_sds; b = g.ll_ours_T .- g.ll_sds
        @printf("%-12s ours(t_last) − SDS: max |Δ| %.3g; ours(T) − SDS: %.3g to %.3g; Δ between points (t_last) %.3g\n",
                g.set[1], maximum(abs.(a)), minimum(b), maximum(b), maximum(a) - minimum(a))
    end

    # ------------------------------------------------------------------ fits
    jobs = vcat([(s, arm) for s in sets if s.known for arm in (:ours_sdsprior, :ours_s3prior, :alg41_sdsprior, :alg41_s4prior)],
                [(s, arm) for s in sets if !s.known for arm in (:alg41_sdsprior, :alg41_s4prior)])
    rows = Vector{Any}(undef, length(jobs))
    Threads.@threads for j in eachindex(jobs)
        s, arm = jobs[j]
        sim, T = load_set(s.file; n = s.n_init)
        de = obs_exact(sim, T); dd = obs_steps(sim, T)
        rng = Xoshiro(hash((s.name, arm)))
        r = arm === :ours_sdsprior ? fit_sdsprior(rng, de, dd) :
            arm === :ours_s3prior  ? merge(fit_bayes(rng, dsa_model(de), de, dd.meanδ, :end; n_draws = 2000), (N = fill(NaN, 5),)) :
            fit_dsa_gibbs(rng, de, dd, :sds, :end, 10^7; th = 1, lprior = arm === :alg41_sdsprior ? logprior_sds : logprior_x)
        rows[j] = (set = s.name, fitn = arm in (:alg41_sdsprior, :alg41_s4prior), impl = string(arm), time_s = r.time,
                   R0_med = r.R0[3], R0_lo = r.R0[4], R0_hi = r.R0[5], mu_med = r.μ[3], mu_lo = r.μ[4], mu_hi = r.μ[5],
                   beta_med = r.β[3], beta_lo = r.β[4], beta_hi = r.β[5], rho_med = r.ρ[3], rho_lo = r.ρ[4], rho_hi = r.ρ[5],
                   N_med = r.N[3], N_lo = r.N[4], N_hi = r.N[5], rhat = r.rhat, ess = r.ess)
        @printf("%-12s %-15s R0 %.3f [%.3f, %.3f]  mu %.3f  N %.0f  R̂ %.3f  %.0f s\n", s.name, arm, r.R0[3], r.R0[4], r.R0[5],
                r.μ[3], r.N[3], r.rhat, r.time)
    end
    CSV.write(joinpath(@__DIR__, "out", "s6_5_sds_own_ours.csv"), DataFrame(rows))
end
