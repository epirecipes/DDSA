# S6: computational cost on an idle machine, single thread. For each arm: median time of one log-likelihood
# evaluation and of one ForwardDiff gradient, and one complete fit (wall time, minimum bulk ESS, ESS per second).
# S3/S4 arms at n ∈ {100, 1,000, 10,000} (R0 = 2, end cut-off, replicate 1); S5 arms on replicate 1 (scenario A).
# Usage (from the supplementary folder, nothing else running): julia --project=@. -t 1 S3_S6_dsa_comparison/s6_1_benchmarks.jl
include(joinpath(@__DIR__, "s4_popsize.jl")); include(joinpath(@__DIR__, "s5_nonexp.jl"))
using ForwardDiff
function tmed(f; n = 20)
    f(); ts = [(@elapsed f()) for _ in 1:n]; median(ts)
end
rows = String[]; say(s) = (println(s); flush(stdout); push!(rows, s))
say("block,arm,n,K,loglik_ms,gradient_ms,fit_s,min_ess_bulk,ess_per_s,rhat")
for n in (100, 1000, 10_000)
    R0 = 2.0; sim = sim_outbreak(Xoshiro(hash(("s3", n, R0, 1))), n, round(Int, 0.05n), R0 / 5.0, Exponential(5.0))
    m = round(Int, 0.05n); mf = meanfield_cutoffs(R0, 5.0, m / (n + m)); T = ceil(mf.t99)
    dd = obs_steps(sim, T); de = obs_exact(sim, T); x0 = [R0 / 5.0, 5.0, m / (n + m)]
    Nmax = 5n; gd = NGrid(dd.K, dd.K + 1, Nmax); ge = NGrid(de.K, de.K, Nmax - m)
    arms = [("S3", "ddsa", x -> ddsa_loglik(x..., dd), () -> fit_bayes(Xoshiro(1), ddsa_model(dd), dd, dd.meanδ, :end)),
            ("S3", "dsa", x -> dsa_loglik(x..., de), () -> fit_bayes(Xoshiro(1), dsa_model(de), de, dd.meanδ, :end)),
            ("S4", "ddsa_N", x -> ddsa_loglik_N(x..., dd, gd), () -> fit_ddsa_N(Xoshiro(1), dd, :end, Nmax)),
            ("S4", "dsa_full", x -> dsa_loglik_N(x..., de, ge), () -> fit_dsa_N(Xoshiro(1), de, dd, :end, Nmax)),
            ("S4", "dsa_sds", x -> dsa_loglik(x..., merge(de, (n = n,))), () -> fit_dsa_gibbs(Xoshiro(1), de, dd, :sds, :end, Nmax))]
    for (blk, arm, f, fit) in arms
        tl = tmed(() -> f(x0)); tg = arm == "dsa_sds" ? NaN : tmed(() -> ForwardDiff.gradient(f, x0))
        r = fit()
        say(join(Any[blk, arm, n, dd.K, round(1000tl, sigdigits = 3), round(1000tg, sigdigits = 3), round(r.time, digits = 1),
                     round(r.ess, digits = 0), round(r.ess / r.time, sigdigits = 3), round(r.rhat, digits = 4)], ","))
    end
end
sim = simulate5(1); T = ceil(maximum(vcat(sim.rec_t, sim.seed_rec))) + 1.0
dd = obs_steps(sim, T); db = obs_steps_box(sim, T); de = obs_exact_w(sim, T); ρt = MS5 / (NS5 + MS5)
rmax = min(R_MAX5, dd.du[1])
for (arm, f, mdl) in (("ddsa_dph", x -> marginal_r_loglik(r -> ddsa_dph_loglik(x[1], x[2], x[3], r, dd, :A), 1:rmax)[1], :ddsa),
                      ("ddsa_box", x -> ddsa_box_loglik(x[1], x[2], K5, x[3], db, :A), :ddsa_box),
                      ("dsa_gamma", x -> dsa_gamma_loglik(x[1], x[2], K5, x[3], de, :A), :dsa))
    x0 = [β5, μ5, ρt]
    tl = tmed(() -> f(x0); n = 5); tg = tmed(() -> ForwardDiff.gradient(f, x0); n = 5)
    r = fit5(Xoshiro(1), mdl, :A, sim, T)
    say(join(Any["S5", arm, NS5, dd.K, round(1000tl, sigdigits = 3), round(1000tg, sigdigits = 3), round(r.time, digits = 1),
                 round(r.ess, digits = 0), round(r.ess / r.time, sigdigits = 3), round(r.rhat, digits = 4)], ","))
end
open(io -> foreach(l -> println(io, l), rows), joinpath(OUTDIR, "s6_benchmarks.csv"), "w")
