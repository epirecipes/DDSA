# Step 3: data drawn from the DDSA working model. The full-data likelihood treats the cases as independent given the
# parameters (Binomial number of cases, infection days drawn from the mean-field incidence, independent DPH intervals).
# Here the data are drawn from exactly that distribution (sim_working_model), at the cells of the chain-binomial grid
# with r = 4 (N ∈ {100, 1000, 10000}, R0 ∈ {1.3, 2, 3}, replicates 1–100), and fitted with the same DPH model, priors,
# sampler and starts. Comparing the coverage with that of the chain-binomial grid separates what the likelihood's
# independence assumption does from the rest of the inference. Resumable; stops starting fits after BUDGET s.
#   INIT=data|prior (default data, as in the grid)
# Usage (from the supplementary folder): julia --project=@. -t 10 S7_stress_tests/baseline_grid/final_step3_working_model.jl
include(joinpath(@__DIR__, "sim_core.jl"))
const INIT = Symbol(get(ENV, "INIT", "data"))
const BUDGET = parse(Float64, get(ENV, "BUDGET", "420"))
const RES = joinpath(OUTDIR, "final_step3_working_model.csv")
const CELLS = [(N, R0) for N in (100, 1000, 10_000) for R0 in (1.3, 2.0, 3.0)]
t0 = time()
dataset(N, R0, r, rep) = linelist(sim_working_model(Xoshiro(hash(("wm", N, R0, r, rep))), N, R0, r), N)

isfile(RES) || open(io -> println(io, "N,R0,rep,K,T,init,time_s,beta_true,beta_med,beta_lo,beta_hi,ip_true,ip_med,ip_lo,ip_hi," *
                                      "R0_true,R0_med,R0_lo,R0_hi,r_med,rhat,ess,div,lpgap"), RES, "w")
done = Set{Tuple}()
for l in Iterators.drop(eachline(RES), 1)
    f = split(l, ","); push!(done, (parse(Int, f[1]), parse(Float64, f[2]), parse(Int, f[3])))
end
jobs = [(N, R0, rep) for rep in 1:100 for (N, R0) in CELLS if !((N, R0, rep) in done)]
println("pending $(length(jobs))")
lk = ReentrantLock(); nxt = Threads.Atomic{Int}(1)
Threads.@threads for _ in 1:Threads.nthreads()
    while time() - t0 < BUDGET
        j = Threads.atomic_add!(nxt, 1); j > length(jobs) && break
        N, R0, rep = jobs[j]
        d = dataset(N, R0, 4, rep)
        res = try fit_one(Xoshiro(hash(("wm-fit", N, R0, rep))), d, :dph; init = INIT) catch e; @warn "fail" e; nothing end
        res === nothing && continue
        lock(lk) do
            open(RES, "a") do io
                println(io, join(Any[N, R0, rep, d.K, d.nsteps, INIT, round(res.time, digits=1), R0 / WBAR, res.β..., WBAR, res.ip...,
                                     R0, res.R0..., res.rmed, round(res.rhat, digits=4), round(res.ess), res.div, round(res.lpgap, digits=2)], ","))
            end
        end
    end
end
n = countlines(RES) - 1
println("fits done: $n of 900; remaining: $(900 - n)")
