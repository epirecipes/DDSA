# Checks for stress_core.jl (log only): (1) with every option at its null value the general generator reproduces
# sim_chainbinomial exactly (same seed); (2) the stress likelihood with N known and censoring off equals the S7 baseline-grid
# full_loglik (sim_core.jl); (3) one quick fit per factor (first level), to time them.
# Usage (from the supplementary folder): julia --project=@. -t 10 S7_stress_tests/stress_check.jl
include(joinpath(@__DIR__, "stress_run.jl"))
say(x...) = (println(x...); flush(stdout))
ok = all(1:200) do i
    a = sim_chainbinomial(Xoshiro(i), N7, R07, r7); b = sim_cb_general(Xoshiro(i), N7, R07, r7)
    a.infday == b.infday && a.recday == b.recday && a.seed_delta == b.seed_delta
end
say("generator null level reproduces sim_chainbinomial for 200 seeds: ", ok)
s = sim_chainbinomial(Xoshiro(11), N7, R07, r7); d0 = linelist(s, N7); d1 = stress_data(Xoshiro(1), s, N7)
for (r, β, ρ, p) in ((4, 0.32, 0.001, 0.6), (3, 0.30, 0.002, 0.5), (1, 0.35, 0.001, 0.16))
    say(@sprintf("r=%d: full_loglik %.8f  stress_loglik %.8f", r, full_loglik(r, β, ρ, p, d0), stress_loglik(r, β, ρ, p, d1)))
end
first_levels = unique(first.(LEVELS)); results = Vector{Any}(undef, length(first_levels))
Threads.@threads for i in eachindex(first_levels)
    f = first_levels[i]; l = LEVELS[findfirst(x -> x[1] == f, LEVELS)][2]
    d, tru, Nmode, att = make_case(f, l, 1)
    r = fit_stress(Xoshiro(1), d; Nmode)
    results[i] = @sprintf("%-18s %-28s K=%4d T=%3d rmax=%d %5.1f s R0 %.3f [%.3f, %.3f] (true %.2f) IP %.2f (true %.2f) r %.0f R̂ %.3f div %d p_inc %.2f p_δ %.2f p_K %.2f",
        f, l, d.K, d.nsteps, r.rmax, r.time, r.R0..., tru.R0, r.ip[1], tru.ip, r.rmed, r.rhat, r.div, r.p_inc, r.p_delta, r.p_K)
end
foreach(say, results)
