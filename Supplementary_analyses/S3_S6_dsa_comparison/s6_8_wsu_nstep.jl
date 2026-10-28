# S6: why DDSA and the SDS analysis of KhudaBukhsh et al. (2020) estimate different population sizes for WSU H1N1.
# Continuous-time DSA on the WSU line list (exact times, T = 104, the 11 cases at t = 0 as initial infectives), with
# our re-implementation of their Algorithm 4.1, which reproduces their SDS R code on these data (s6_4/s6_5), and the
# priors of their example1.R (logprior_sds). Only the treatment of n changes between the arms:
#   alg41    n − K ~ NegBin(K, tau_inf), likelihood without C(n, K) (their Algorithm 4.1)
#   full     n − K ~ NegBin(K + 1, q_T), q_T = 1 − s_T/s_0: the exact conditional of n under the complete Binomial
#            final-size term with a uniform prior on n, which adds C(n, K); same Metropolis-within-Gibbs sampler
#   summed   n summed out of the same complete Binomial term, NUTS (S3 priors), as DDSA does for H1N1
# The full and summed arms use the bound of the DDSA H1N1 notebook, N ≤ 18,000.
# Usage (from the supplementary folder): julia --project=@. -t 3 S3_S6_dsa_comparison/s6_8_wsu_nstep.jl
include(joinpath(@__DIR__, "s6_5_sds_own_settings.jl"))   # loaders and logprior_sds (its own runs execute only as the main script)

const NMAX_WSU = 18_000
sim, T = load_set(joinpath(DATA, "sdsown_wsu.csv"); n = 7803)
de = obs_exact(sim, T); dd = obs_steps(sim, T)
arms = (:alg41, :full, :summed)
res = Vector{Any}(undef, length(arms))
Threads.@threads for j in eachindex(arms)
    a = arms[j]; rng = Xoshiro(hash(("s6_8", a)))
    res[j] = a === :alg41 ? fit_dsa_gibbs(rng, de, dd, :sds, :end, NMAX_WSU; th = 1, lprior = logprior_sds) :
             a === :full  ? fit_dsa_gibbs(rng, de, dd, :full, :end, NMAX_WSU; th = 5, lprior = logprior_sds) :
                            fit_dsa_N(rng, de, dd, :end, NMAX_WSU)
end
rows = [(arm = string(a), R0_med = r.R0[3], R0_lo = r.R0[4], R0_hi = r.R0[5], mu_med = r.μ[3], beta_med = r.β[3],
         N_med = r.N[3], N_lo = r.N[4], N_hi = r.N[5], rhat = r.rhat, ess = r.ess, time_s = r.time) for (a, r) in zip(arms, res)]
CSV.write(joinpath(@__DIR__, "out", "s6_8_wsu_nstep.csv"), DataFrame(rows))
for r in rows
    @printf("%-7s R0 %.3f [%.3f, %.3f]  mean IP %.2f d  beta %.3f  N %.0f [%.0f, %.0f]  R̂ %.3f  ESS %.0f  %.0f s\n",
            r.arm, r.R0_med, r.R0_lo, r.R0_hi, r.mu_med, r.beta_med, r.N_med, r.N_lo, r.N_hi, r.rhat, r.ess, r.time_s)
end
