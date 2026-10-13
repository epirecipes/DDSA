# Supplement S5, step refinement of the Boxcar arm: tests whether the +3.8% R0 bias of DDSA with the Boxcar structure and a
# discretised Gamma infectious period (scenario A, Table S15) comes from the daily step of the map (M); in S5 it falls to
# +1.8% at Δt = 0.5 d and +0.8% at 0.25 d.
# Same outbreaks (simulate5), priors, sampler and starts as s5_nonexp.jl; only the resolution changes. Records at
# resolution Δt (as the Δt arms of s3_1_run.jl): infection step k = ⌊t/Δt⌋ + 1, interval δ = max(1, recovery step − k),
# seeds δ = ⌊r/Δt⌋ + 1; the map runs on steps of Δt days with infection probability 1 − exp(−β I Δt), and the interval
# law is the discretised Gamma in step units (mean μ/Δt steps), i.e. the recording convention of s5_nonexp.jl at
# resolution Δt. At Δt = 1 the likelihood equals ddsa_box_loglik exactly (checked at start-up).
#   DTS=0.5,0.25  REPS=20  SCEN=A  BUDGET=540  RUN=s5_boxcar_dt
# Usage (from the supplementary folder): DTS=0.5 REPS=2 RUN=s5_boxcar_dt_pilot julia --project=@. -t 8 S3_S6_dsa_comparison/s5_boxcar_dt.jl
include(joinpath(@__DIR__, "s5_nonexp.jl"))

"""obs_steps at resolution Δt with the seeds' intervals separated (no censoring: outbreaks run to extinction)."""
function obs_steps_box_dt(sim, T, Δt)
    dd = obs_steps(sim, T; Δt = Δt)
    sd = [floor(Int, r / Δt) + 1 for r in sim.seed_rec if r < T]
    cd = [max(1, floor(Int, r / Δt) + 1 - (floor(Int, t / Δt) + 1)) for (t, r) in zip(sim.inf_t, sim.rec_t) if t < T && r < T]
    du = sort(unique(cd)); dn = [count(==(x), cd) for x in du]
    su = sort(unique(sd)); sn = [count(==(x), sd) for x in su]
    return merge(dd, (du = du, dn = dn, su = su, sn = sn))
end

"""ddsa_box_loglik with the interval law in steps of Δt (mean μ/Δt steps); identical to it at Δt = 1."""
function ddsa_box_loglik_dt(β, μ, k, ρ, d, use)
    (0 < ρ < 1 && μ > 0 && k >= 1) || return -Inf
    T_ = promote_type(typeof(β), typeof(μ), typeof(k), typeof(ρ))
    A = max(d.nsteps, maximum(vcat(d.du, d.su))) + 1
    pc, ps = discretised_gamma(μ / d.Δt, k, A)
    ll = zero(T_)
    if use !== :C
        Sc = reverse(cumsum(reverse(pc)))
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
@model function ddsa_box_model_dt(d, use)
    β ~ Uniform(β_LO, β_HI); μ ~ Uniform(μ_LO, μ_HI); k ~ Uniform(k_LO, k_HI); ρ ~ Beta(ρ_A, ρ_B)
    Turing.@addlogprob! ddsa_box_loglik_dt(β, μ, k, ρ, d, use)
end

function check_dt1()
    sim = simulate5(1); T = ceil(maximum(vcat(sim.rec_t, sim.seed_rec))) + 1.0
    d0 = obs_steps_box(sim, T); d1 = obs_steps_box_dt(sim, T, 1.0)
    rng = Xoshiro(7); mx = 0.0
    for _ in 1:20, use in (:A, :B, :C)
        β = 0.15 + 0.2rand(rng); μ = 6 + 6rand(rng); k = 1 + 25rand(rng); ρ = 0.001 + 0.01rand(rng)
        mx = max(mx, abs(ddsa_box_loglik(β, μ, k, ρ, d0, use) - ddsa_box_loglik_dt(β, μ, k, ρ, d1, use)))
    end
    println("check Δt = 1: max |ddsa_box_loglik − ddsa_box_loglik_dt| over 60 points = $mx"); flush(stdout)
    mx == 0.0 || error("Δt = 1 variant does not reproduce ddsa_box_loglik")
end

function fit_box_dt(rng, use, sim, T, Δt)
    db = obs_steps_box_dt(sim, T, Δt)
    vals = init_values(rng, db, db.meanδ, :end)
    w = vcat(fill.(db.du .* Δt, db.dn)...)                           # recorded intervals in days
    k0 = clamp(db.meanδ^2 / max(var(w) - Δt^2 / 6, 0.2), 1.0, 25.0)  # minus rounding variance
    inits = [InitFromParams(merge(v, (k = clamp(k0 * exp(0.1 * randn(rng)), 1.05, 29.0),))) for v in vals]
    m = ddsa_box_model_dt(db, use)
    t = @elapsed ch = sample(rng, m, NUTS(1000, 0.8; max_depth = 12, Δ_max = 1000.0), MCMCSerial(), 1000, 4;
                             progress = false, chain_type = MCMCChains.Chains, initial_params = inits)
    β = vec(ch[:β]); μ = vec(ch[:μ]); ρ = vec(ch[:ρ]); kk = vec(ch[:k])
    pars = [:β, :μ, :k, :ρ]; s = summarystats(ch[pars])
    arr = permutedims(cat([Array(ch[:, pars, c]) for c in 1:4]...; dims = 3), (1, 3, 2))
    return (db = db, time = t, β = summ(β), μ = summ(μ), R0 = summ(β .* μ), ρ = summ(ρ), sd = summ(μ ./ sqrt.(kk)),
            shape = summ(kk), rhat = maximum(s[:, :rhat]), ess = minimum(s[:, :ess_bulk]),
            ess_tail = minimum(MCMCDiagnosticTools.ess(arr; kind = :tail)), div = Int(sum(vec(ch[:numerical_error]))))
end

if abspath(PROGRAM_FILE) == @__FILE__
    check_dt1()
    DTS = parse.(Float64, split(get(ENV, "DTS", "0.5,0.25"), ","))
    REPS = parse(Int, get(ENV, "REPS", "20")); BUDGET = parse(Float64, get(ENV, "BUDGET", "540"))
    SCEN = Symbol.(split(get(ENV, "SCEN", "A"), ","))
    RUN = get(ENV, "RUN", "s5_boxcar_dt"); RES = joinpath(OUTDIR, "$RUN.csv"); PARS = (:β, :μ, :R0, :ρ, :sd, :shape)
    HDR = join(vcat(["rep", "scenario", "model", "dt", "K", "T", "nsteps", "time_s"],
                    [string(p, "_", s) for p in PARS for s in ("true", "mean", "sd", "med", "lo", "hi")],
                    ["rhat", "ess", "ess_tail", "div"]), ",")
    t0 = time()
    isfile(RES) || open(io -> println(io, HDR), RES, "w")
    done = Set{Tuple}()
    for l in Iterators.drop(eachline(RES), 1)
        f = split(l, ","); push!(done, (parse(Int, f[1]), Symbol(f[2]), parse(Float64, f[4])))
    end
    # coarser steps first (cheaper), so a budget stop leaves complete Δt levels
    jobs = [(rep, use, dt) for dt in sort(DTS; rev = true) for use in SCEN for rep in 1:REPS if !((rep, use, dt) in done)]
    println("pending jobs: $(length(jobs)) (done: $(length(done)))"); flush(stdout)
    lk = ReentrantLock(); next = Threads.Atomic{Int}(1); nfit = Threads.Atomic{Int}(0)
    Threads.@threads for _ in 1:Threads.nthreads()
        while true
            time() - t0 > BUDGET && break
            j = Threads.atomic_add!(next, 1); j > length(jobs) && break
            rep, use, dt = jobs[j]
            sim = simulate5(rep); T = ceil(maximum(vcat(sim.rec_t, sim.seed_rec))) + 1.0
            res = try
                fit_box_dt(Xoshiro(hash(("s5fit", rep, use, :ddsa_box, dt))), use, sim, T, dt)
            catch e
                @warn "fit failed" rep use dt e; nothing
            end
            res === nothing && continue
            tru = (β = β5, μ = μ5, R0 = β5 * μ5, ρ = MS5 / (NS5 + MS5), sd = sqrt(V5), shape = K5)
            row = join(Any[rep, use, :ddsa_box, dt, res.db.K, T, res.db.nsteps, round(res.time, digits = 1),
                           vcat([[tru[p], getfield(res, p)...] for p in PARS]...)...,
                           round(res.rhat, digits = 4), round(res.ess, digits = 0), round(res.ess_tail, digits = 0), res.div], ",")
            lock(lk) do; open(io -> println(io, row), RES, "a"); end
            Threads.atomic_add!(nfit, 1)
            println("rep $rep $use Δt=$dt: R0 med $(round(res.R0[3], digits=3)) | $(round(res.time, digits=0)) s"); flush(stdout)
        end
    end
    println("fits completed this run: $(nfit[]) in $(round(time() - t0, digits=0)) s; remaining: $(length(jobs) - nfit[])")
end
