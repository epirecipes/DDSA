# Resumable runner for the S7 baseline grid. Each invocation fits pending (cell, replicate, model) jobs in parallel
# (one fit per thread, 4 serial chains each) and stops starting new jobs after BUDGET seconds, so a run stays within the
# 10-minute cap; completed fits are appended to out/results_<SIM>.csv and skipped on the next invocation.
#   SIM=cb|gil   chain-binomial (default) or Gillespie;   REPS=100 replicates per cell (Gillespie default 20)
#   BUDGET=420   seconds after which no new fit is started
# Usage (from the supplementary folder): SIM=cb REPS=10 julia --project=@. -t 10 S7_stress_tests/baseline_grid/sim_run.jl
include(joinpath(@__DIR__, "sim_core.jl"))
const SIM    = get(ENV, "SIM", "cb")
const REPS   = parse(Int, get(ENV, "REPS", SIM == "cb" ? "100" : "20"))
const BUDGET = parse(Float64, get(ENV, "BUDGET", "420"))
const CELLS  = [(N, R0, r) for N in (100, 1000, 10_000) for R0 in (1.3, 2.0, 3.0) for r in (1, 4)]
const RES    = joinpath(OUTDIR, "results_$SIM.csv")
const DSET   = joinpath(OUTDIR, "datasets_$SIM.csv")
const HDR    = "sim,N,R0,r,rep,model,attempts,K,T,time_s,beta_true,beta_med,beta_lo,beta_hi,ip_true,ip_med,ip_lo,ip_hi,R0_true,R0_med,R0_lo,R0_hi,r_med,rhat,ess,div"
t0 = time()

"""Dataset for (cell, rep): simulate until a major outbreak (K ≥ 0.1 N); deterministic seed; returns data and attempts."""
function dataset(N, R0, r, rep)
    base = hash((SIM, N, R0, r, rep))
    attempts = 0
    while true
        attempts += 1
        rng = Xoshiro(base + UInt(attempts))
        s = SIM == "cb" ? sim_chainbinomial(rng, N, R0, r) : sim_gillespie(rng, N, R0, r)
        s.K >= 0.1N && return linelist(s, N), attempts
        attempts > 10_000 && error("no major outbreak in 10,000 attempts")
    end
end

isfile(RES) || open(io -> println(io, HDR), RES, "w")
done = Set{Tuple}()
for l in Iterators.drop(eachline(RES), 1)
    f = split(l, ","); push!(done, (parse(Int, f[2]), parse(Float64, f[3]), parse(Int, f[4]), parse(Int, f[5]), Symbol(f[6])))
end
# job order: replicate-major, so a partial run covers every cell. For r = 1 the DPH model has r_max = min δ, which is 1
# (so the DPH model is the geometric model) in all but one of the 900 r = 1 datasets (2 in one at N = 100, R0 = 1.3);
# only one fit is made and recorded under both names.
jobs = [(N, R0, r, rep, m) for rep in 1:REPS for (N, R0, r) in CELLS for m in (r == 1 ? (:dph,) : (:dph, :geo))
        if !((N, R0, r, rep, m) in done)]
println("pending jobs: $(length(jobs)) (done: $(length(done)))")
lk = ReentrantLock(); next = Threads.Atomic{Int}(1); nfit = Threads.Atomic{Int}(0)
Threads.@threads for _ in 1:Threads.nthreads()
    while true
        time() - t0 > BUDGET && break
        j = Threads.atomic_add!(next, 1); j > length(jobs) && break
        N, R0, r, rep, m = jobs[j]
        d, att = dataset(N, R0, r, rep)
        res = try
            fit_one(Xoshiro(hash((SIM, N, R0, r, rep, m))), d, m)
        catch e
            @warn "fit failed" N R0 r rep m e; nothing
        end
        res === nothing && continue
        row(model) = join(Any[SIM, N, R0, r, rep, model, att, d.K, d.nsteps, round(res.time, digits=1),
                              R0 / WBAR, res.β..., WBAR, res.ip..., R0, res.R0..., res.rmed, round(res.rhat, digits=4),
                              round(res.ess, digits=0), res.div], ",")
        lock(lk) do
            open(RES, "a") do io
                println(io, row(m)); r == 1 && println(io, row(:geo))
            end
        end
        Threads.atomic_add!(nfit, 1)
    end
end
println("fits completed this run: $(nfit[]) in $(round(time() - t0, digits=0)) s; remaining: $(length(jobs) - nfit[])")
