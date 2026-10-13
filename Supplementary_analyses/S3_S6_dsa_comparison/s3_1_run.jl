# Resumable runner for the S3 comparison. Each invocation fits pending jobs in parallel (one fit per thread, 4 serial
# chains) and stops starting new jobs after BUDGET seconds; finished fits are appended to out/<RUN>.csv and skipped next time.
#   REPS=50       replicates per cell          BUDGET=420   seconds after which no new fit starts
#   RUN=s3_main   output name (s3_pilot for the pilot)
# Cells: n ∈ {100, 1,000, 10,000} × R0 ∈ {1.5, 2, 4} × cut-off ∈ {peak, end}; mean IP 5 d, ρ_S = 0.05 (KhudaBukhsh et al.).
# Models per cell: ddsa (daily records), dsa_exact (exact times), dsa_jitter (daily records spread within the day), exact
# (complete-data benchmark); plus ddsa at Δt = 0.5, 0.25, 0.1 d for n = 1,000 at the end cut-off (convergence check).
# Usage (from the supplementary folder): REPS=5 RUN=s3_pilot julia --project=@. -t 10 S3_S6_dsa_comparison/s3_1_run.jl
include(joinpath(@__DIR__, "s3_core.jl"))
const REPS   = parse(Int, get(ENV, "REPS", "50"))
const BUDGET = parse(Float64, get(ENV, "BUDGET", "420"))
const RUN    = get(ENV, "RUN", "s3_main")
const MU     = 5.0
const RHO_S  = 0.05
const NS     = (100, 1000, 10_000)
const R0S    = (1.5, 2.0, 4.0)
const RES    = joinpath(OUTDIR, "$RUN.csv")
const PARS   = (:β, :μ, :R0, :ρ)
const HDR    = join(vcat(["n", "R0", "cut", "rep", "model", "dt", "K", "T", "nrec", "time_s"],
                         [string(p, "_", s) for p in PARS for s in ("true", "mean", "sd", "med", "lo", "hi")],
                         ["rhat", "ess", "ess_tail", "div", "lpgap"]), ",")
t0 = time()

cutoffs(n, R0) = (mf = meanfield_cutoffs(R0, MU, round(Int, RHO_S * n) / (n + round(Int, RHO_S * n)));
                  (peak = ceil(mf.tpeak), stop = ceil(mf.t99)))
simulate(n, R0, rep) = sim_outbreak(Xoshiro(hash(("s3", n, R0, rep))), n, round(Int, RHO_S * n), R0 / MU, Exponential(MU))

isfile(RES) || open(io -> println(io, HDR), RES, "w")
done = Set{Tuple}()
for l in Iterators.drop(eachline(RES), 1)
    f = split(l, ","); push!(done, (parse(Int, f[1]), parse(Float64, f[2]), Symbol(f[3]), parse(Int, f[4]), Symbol(f[5]), parse(Float64, f[6])))
end
jobs = Tuple[]
for rep in 1:REPS, n in NS, R0 in R0S, cut in (:peak, :end)
    for (mdl, dt) in ((:exact, 0.0), (:ddsa, 1.0), (:dsa_exact, 0.0), (:dsa_jitter, 1.0))
        push!(jobs, (n, R0, cut, rep, mdl, dt))
    end
    if n == 1000 && cut === :end
        for dt in (0.5, 0.25, 0.1); push!(jobs, (n, R0, cut, rep, :ddsa, dt)); end
    end
end
filter!(j -> !(j in done), jobs)
sort!(jobs, by = j -> (j[4], -j[1]))          # replicate-major; within a replicate the slow (large n) fits first
println("pending jobs: $(length(jobs)) (done: $(length(done)))"); flush(stdout)

lk = ReentrantLock(); next = Threads.Atomic{Int}(1); nfit = Threads.Atomic{Int}(0)
Threads.@threads for _ in 1:Threads.nthreads()
    while true
        time() - t0 > BUDGET && break
        j = Threads.atomic_add!(next, 1); j > length(jobs) && break
        n, R0, cut, rep, mdl, dt = jobs[j]
        sim = simulate(n, R0, rep); co = cutoffs(n, R0); T = cut === :peak ? co.peak : co.stop
        rng = Xoshiro(hash(("s3fit", n, R0, cut, rep, mdl, dt)))
        dd = obs_steps(sim, T; Δt = mdl === :ddsa ? dt : 1.0)
        res, d = try
            if mdl === :exact
                fit_exact(rng, sim, T), obs_exact(sim, T)
            elseif mdl === :ddsa
                fit_bayes(rng, ddsa_model(dd), dd, dd.meanδ, cut), dd
            elseif mdl === :dsa_exact
                de = obs_exact(sim, T); fit_bayes(rng, dsa_model(de), de, dd.meanδ, cut), de
            else
                dj = obs_jitter(rng, sim, T); fit_bayes(rng, dsa_model(dj), dj, dd.meanδ, cut), dj
            end
        catch e
            @warn "fit failed" n R0 cut rep mdl dt e; (nothing, nothing)
        end
        res === nothing && continue
        m = sim.m; tru = (β = R0 / MU, μ = MU, R0 = R0, ρ = m / (n + m))
        nrec = mdl === :ddsa ? sum(dd.dn) : d.nrec
        row = join(Any[n, R0, cut, rep, mdl, dt, d.K, T, nrec, round(res.time, digits = 1),
                       vcat([[tru[p], getfield(res, p)...] for p in PARS]...)...,
                       round(res.rhat, digits = 4), round(res.ess, digits = 0), round(res.ess_tail, digits = 0), res.div,
                       round(res.lpgap, digits = 2)], ",")
        lock(lk) do
            open(io -> println(io, row), RES, "a")
        end
        Threads.atomic_add!(nfit, 1)
    end
end
println("fits completed this run: $(nfit[]) in $(round(time() - t0, digits=0)) s; remaining: $(length(jobs) - nfit[])")
