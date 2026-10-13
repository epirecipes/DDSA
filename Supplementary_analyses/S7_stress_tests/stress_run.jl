# Resumable runner for the S7 stress tests (one fit per thread, 4 serial chains; stops starting fits after BUDGET s).
#   REPS=50  BUDGET=420  RUN=stress_main
# Usage (from the supplementary folder): REPS=5 RUN=stress_pilot julia --project=@. -t 10 S7_stress_tests/stress_run.jl
include(joinpath(@__DIR__, "stress_core.jl"))
const REPS   = parse(Int, get(ENV, "REPS", "50"))
const BUDGET = parse(Float64, get(ENV, "BUDGET", "420"))
const RUN    = get(ENV, "RUN", "stress_main")
const RES    = joinpath(OUT7, "$RUN.csv")

# factor => levels (the baseline is the null level of every factor). No commas in names: they are CSV fields.
const LEVELS = [
    ("baseline", "baseline"),
    ("resolution", "IP 2 d"), ("resolution", "IP 4 d"), ("resolution", "IP 6.33 d"), ("resolution", "IP 12 d"),
    ("misrecorded", "1%"), ("misrecorded", "5%"),
    ("missing_recovery", "MCAR 30%"), ("missing_recovery", "MCAR 60%"), ("missing_recovery", "long intervals 30%"),
    ("window", "peak; censored"), ("window", "peak; dropped"), ("window", "peak; truncation term"),
    ("window", "early; censored"), ("window", "early; dropped"), ("window", "early; truncation term"),
    ("ascertainment", "70%"), ("ascertainment", "40%"),
    ("unknown_N", "fraction-scale seed prior"), ("unknown_N", "count-scale seed prior"),
    ("beta_t", "halved at peak"), ("beta_t", "linear decline to 50%"),
    ("mixing", "5 out-of-phase subpopulations"), ("mixing", "susceptibility CV 0.5"), ("mixing", "susceptibility CV 1"),
    ("superspreading", "k = 0.5"), ("superspreading", "k = 0.1"),
    ("importations", "1 per week; days 7-91"),
    ("near_critical", "R0 1.1; N 100"), ("near_critical", "R0 1.1; N 1000"),
]

"""Outbreak for (factor, level, rep): simulate until a major outbreak (≥ 10% of N infected); returns the data, the
truth and the fitting options."""
function make_case(factor, level, rep)
    N = N7; R0 = R07; W̄t = WBAR; opts = (;); Nmode = :known
    factor == "near_critical" && (R0 = 1.1; N = level == "R0 1.1; N 100" ? 100 : 1000)
    if factor == "resolution"
        W̄t = Dict("IP 2 d" => 2.0, "IP 4 d" => 4.0, "IP 6.33 d" => WBAR, "IP 12 d" => 12.0)[level]
    end
    attempts = 0
    while true
        attempts += 1
        rng = Xoshiro(hash(("s7s", factor, level, rep, attempts)))
        s = if factor == "resolution"
            sim_gillespie(rng, N, R0, r7; μ = W̄t)
        elseif factor == "beta_t"
            pk = 45                                                  # day of peak incidence of the baseline mean field
            f = level == "halved at peak" ? (k -> k < pk ? R0 / WBAR : 0.5R0 / WBAR) :
                                            (k -> R0 / WBAR * max(0.5, 1 - 0.5k / (2pk)))
            sim_cb_general(rng, N, R0, r7; βfun = f)
        elseif factor == "mixing" && level == "5 out-of-phase subpopulations"
            sim_subpops(rng, N, R0, r7)
        elseif factor == "mixing"
            cv = level == "susceptibility CV 0.5" ? 0.5 : 1.0
            sim_cb_general(rng, N, R0, r7; susc = rand(rng, Gamma(1 / cv^2, cv^2), N))
        elseif factor == "superspreading"
            sim_cb_general(rng, N, R0, r7; kdisp = level == "k = 0.5" ? 0.5 : 0.1)
        elseif factor == "importations"
            sim_cb_general(rng, N, R0, r7; imports = collect(7:7:91))
        else
            sim_chainbinomial(rng, N, R0, r7)
        end
        s.K >= 0.1N || (attempts > 20_000 && error("no major outbreak"); continue)
        T = s.T; window = :censor; thin = 1.0; miss = 0.0; misrec = 0.0
        if factor == "window"
            h = zeros(Int, s.T); for k in s.infday; h[k] += 1; end
            ma = [mean(h[max(1, k - 3):min(end, k + 3)]) for k in 1:s.T]
            T = startswith(level, "peak") ? argmax(ma) : findfirst(>=(0.15 * s.K), cumsum(h))
            window = endswith(level, "censored") ? :censor : endswith(level, "dropped") ? :drop : :trunc
        end
        factor == "ascertainment" && (thin = level == "70%" ? 0.7 : 0.4)
        factor == "missing_recovery" && (miss = level == "MCAR 30%" ? 0.3 : level == "MCAR 60%" ? 0.6 : :long)
        factor == "misrecorded" && (misrec = level == "1%" ? 0.01 : 0.05)
        factor == "unknown_N" && (Nmode = level == "fraction-scale seed prior" ? :fraction : :count)
        d = stress_data(rng, s, N; T, thin, miss, misrec, window)
        return d, (R0 = R0, ip = W̄t, β = R0 / W̄t), Nmode, attempts
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    HDR = "factor,level,rep,attempts,K,T,rmax,time_s,R0_true,R0_med,R0_lo,R0_hi,R0_sd,ip_true,ip_med,ip_lo,ip_hi,ip_sd," *
          "beta_true,beta_med,beta_lo,beta_hi,r_med,rhat,ess,ess_tail,div,p_inc,p_delta,p_K"
    t0 = time()
    isfile(RES) || open(io -> println(io, HDR), RES, "w")
    done = Set{Tuple{String,String,Int}}()
    for l in Iterators.drop(eachline(RES), 1)
        f = split(l, ","); push!(done, (String(f[1]), String(f[2]), parse(Int, f[3])))
    end
    jobs = [(f, l, rep) for rep in 1:REPS for (f, l) in LEVELS if !((f, l, rep) in done)]
    println("pending jobs: $(length(jobs)) (done: $(length(done)))"); flush(stdout)
    lk = ReentrantLock(); next = Threads.Atomic{Int}(1); nfit = Threads.Atomic{Int}(0)
    Threads.@threads for _ in 1:Threads.nthreads()
        while true
            time() - t0 > BUDGET && break
            j = Threads.atomic_add!(next, 1); j > length(jobs) && break
            f, l, rep = jobs[j]
            res, d, tru, att = try
                d, tru, Nmode, att = make_case(f, l, rep)
                fit_stress(Xoshiro(hash(("s7sfit", f, l, rep))), d; Nmode), d, tru, att
            catch e
                @warn "fit failed" f l rep e; (nothing, nothing, nothing, 0)
            end
            res === nothing && continue
            row = join(Any[f, l, rep, att, d.K, d.nsteps, res.rmax, round(res.time, digits = 1),
                           tru.R0, res.R0..., res.sd_R0, tru.ip, res.ip..., res.sd_ip, tru.β, res.β..., res.rmed,
                           round(res.rhat, digits = 4), round(res.ess, digits = 0), round(res.ess_tail, digits = 0), res.div,
                           res.p_inc, res.p_delta, res.p_K], ",")
            lock(lk) do; open(io -> println(io, row), RES, "a"); end
            Threads.atomic_add!(nfit, 1)
        end
    end
    println("fits completed this run: $(nfit[]) in $(round(time() - t0, digits=0)) s; remaining: $(length(jobs) - nfit[])")
end
