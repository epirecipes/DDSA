# Step 2 (pre-specified): term attribution. Chain-binomial, r = 4, N = 1000, R0 ∈ {1.3, 2, 3}, replicates 1–100 (300 datasets),
# DPH model with the final code, three variants: baseline (Binomial final size), no final-size term, Normal final size with
# the corrected (Andersson & Britton) variance. Resumable; stops starting fits after BUDGET s.
#   INIT=prior|data (from the step 1 decision)
# Usage (from the supplementary folder): INIT=prior julia --project=@. -t 10 S7_stress_tests/baseline_grid/final_step2_term_attribution.jl
include(joinpath(@__DIR__, "sim_core.jl"))
const SIM = "cb"
const INIT = Symbol(get(ENV, "INIT", "prior"))
const BUDGET = parse(Float64, get(ENV, "BUDGET", "420"))
const RES = joinpath(OUTDIR, "final_step2_term_attribution.csv")
t0 = time()
function dataset(N, R0, r, rep)
    base = hash((SIM, N, R0, r, rep)); attempts = 0
    while true
        attempts += 1
        s = sim_chainbinomial(Xoshiro(base + UInt(attempts)), N, R0, r)
        s.K >= 0.1N && return linelist(s, N), attempts
    end
end
isfile(RES) || open(io -> println(io, "variant,R0,rep,K,init,time_s,beta_true,beta_med,beta_lo,beta_hi,ip_med,ip_lo,ip_hi,R0_true,R0_med,R0_lo,R0_hi,rhat,ess,div,lpgap"), RES, "w")
done = Set{Tuple}()
for l in Iterators.drop(eachline(RES), 1)
    f = split(l, ","); push!(done, (Symbol(f[1]), parse(Float64, f[2]), parse(Int, f[3])))
end
jobs = [(v, R0, rep) for rep in 1:100 for R0 in (1.3, 2.0, 3.0) for v in (:binom, :none, :normal) if !((v, R0, rep) in done)]
println("pending $(length(jobs))")
lk = ReentrantLock(); nxt = Threads.Atomic{Int}(1)
Threads.@threads for _ in 1:Threads.nthreads()
    while time() - t0 < BUDGET
        j = Threads.atomic_add!(nxt, 1); j > length(jobs) && break
        v, R0, rep = jobs[j]
        d, _ = dataset(1000, R0, 4, rep)
        res = try fit_one(Xoshiro(hash(("termattr", v, R0, rep))), d, :dph; init = INIT, fsize = v) catch e; @warn "fail" e; nothing end
        res === nothing && continue
        lock(lk) do
            open(RES, "a") do io
                println(io, join(Any[v, R0, rep, d.K, INIT, round(res.time, digits=1), R0 / WBAR, res.β..., res.ip..., R0, res.R0...,
                                     round(res.rhat, digits=4), round(res.ess), res.div, round(res.lpgap, digits=2)], ","))
            end
        end
    end
end
n = countlines(RES) - 1
println("fits done: $n of 900; remaining: $(900 - n)")
