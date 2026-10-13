# Supplement S2: corroborations. TASK=wide (R0 ~ U(0.5, 20), rungs 1–6); TASK=misc (identities, truncation, seeds,
# never-infected cell); TASK=orig (Section 3(b) count model with its priors swapped for the ladder's).
# WIDE_RUNGS=1,2 restricts TASK=wide to those rungs (to keep each run under 10 min).
# Usage (from the supplementary folder): TASK=wide|misc|orig julia --project=@. -t 4 S2_identifiability/s2_2_corroborations.jl
# Follow-up checks on the identifiability ladder. TASK = wide | misc | orig
include(joinpath(@__DIR__, "s2_model.jl"))       # data, forward maps, likelihoods, `ladder` model
using Serialization

const TASK = ENV["TASK"]
const OUT = open(joinpath(RES, "s2_2_corroborations_$(TASK)$(haskey(ENV, "WIDE_RUNGS") ? "_rungs" * replace(ENV["WIDE_RUNGS"], "," => "") : "").txt"), "w")
say(x) = (println(x); println(OUT, x); flush(OUT))

nuts_std = NUTS(1000, 0.8; max_depth = 12, Δ_max = 1000.0)
function fit(tag, model; seed = 1234, sampler = nuts_std, n_draws = 4000)
    Random.seed!(seed)
    t = @elapsed ch = sample(model, sampler, MCMCThreads(), n_draws, 4; progress = false,
                             chain_type = MCMCChains.Chains, initial_params = fill(InitFromPrior(), 4))
    serialize(joinpath(CHAINS, "$tag.jls"), ch)
    return ch, t
end

function diagstr(ch)
    pars = filter(p -> p in (:R0, :mean_ip, :ρ, :ϕ_, :β, :γ, :i0, :ϕ), names(ch))
    s = summarystats(ch[pars])
    "R̂≤$(round(maximum(s[:, :rhat]), digits=3)) ESS≥$(round(Int, minimum(s[:, :ess_bulk]))) div=$(Int(sum(vec(ch[:numerical_error]))))"
end

# Summary for `ladder`-type models (r summed out); returns r draws
function summ_ladder(name, model, ch, t)
    rd = [sample_index(l) for l in returned(model, ch)]
    R0, ip, ρv = vec(ch[:R0]), vec(ch[:mean_ip]), vec(ch[:ρ])
    τd = [sum(mu_discrete(rd[j], R0[j], ip[j], ρv[j])[1]) / N for j in eachindex(rd)]
    Pr = round.([mean(rd .== r) for r in 1:r_max], digits = 2)
    tail = "P(R0>5)=$(round(mean(R0 .> 5), digits=3))  P(R0>10)=$(round(mean(R0 .> 10), digits=3))"
    say("$name | R0=$(q(R0)) | meanIP=$(q(ip)) d | τ(discrete map)=$(q(τd)) | $tail | P(r)=$Pr | $(diagstr(ch)) | $(round(Int, t)) s")
    return rd
end

if TASK == "wide"
    # ---------------------------------------------------------- 1. wider R0 prior, rungs 1-6
    R0_HI[] = 20.0
    say("R0 ~ Uniform(0.5, 20); all else as the ladder")
    for (i, (name, obs, map, rec)) in enumerate([
            ("1 DDSA Scenario B (discrete, Mult+Binom)", :multbin, :discrete, false),
            ("2 Discrete map, Poisson counts",           :pois,    :discrete, false),
            ("3 Discrete map, NegBin counts",            :nb,      :discrete, false),
            ("4 Continuous ODE, Poisson counts",         :pois,    :ode,      false),
            ("5 Continuous ODE, NegBin counts",          :nb,      :ode,      false),
            ("6 DDSA full line list (Scenario A)",       :multbin, :discrete, true)])
        haskey(ENV, "WIDE_RUNGS") && !(i in parse.(Int, split(ENV["WIDE_RUNGS"], ","))) && continue   # split into capped runs
        m = ladder(map, obs, rec)
        ch, t = fit("wide_rung$i", m)
        summ_ladder(name, m, ch, t)
    end

elseif TASK == "misc"
    # ---------------------------------------------------------- 5a. identity: never-infected cell vs rung 1
    say("== Identity: Multinomial(N; τf, 1−τ) on (h, N−K)  vs  rung 1 (Binomial(N,τ) + Multinomial(K; f))")
    for (r, R0, μ, ρ) in ((1, 3.1, 8.0, 0.01), (4, 2.5, 5.0, 0.02), (7, 4.6, 12.0, 0.005), (5, 3.2, 6.5, 0.01))
        a = ll_rung(r, R0, μ, ρ, 0.0; map = :discrete, obs = :multcell, recov = false)
        b = ll_rung(r, R0, μ, ρ, 0.0; map = :discrete, obs = :multbin,  recov = false)
        say("  r=$r R0=$R0 μ=$μ ρ=$ρ: cell = $a, rung 1 = $b, diff = $(a - b)")
    end

    # ---------------------------------------------------------- 2. truncation at T = 40 (rungs 1, 2, 6; R0 ≤ 5)
    function glct_traj(r, R0, μ, ρ, T)
        dph = PhaseTypeDistributions.dph_negative_binomial(r, r / μ)
        S = 1 - ρ; x = dph.α .* ρ
        for _ in 1:T
            ni = (1 - exp(-(R0 / μ) * sum(x))) * S
            x = dph.S' * x + dph.α .* ni; S -= ni
        end
        return S, sum(x)
    end
    function trunc_report(name, model, ch)
        rd = [sample_index(l) for l in returned(model, ch)]
        R0, ip, ρv = vec(ch[:R0]), vec(ch[:mean_ip]), vec(ch[:ρ])
        rm = round(Int, median(rd)); R0m, ipm, ρm = median(R0), median(ip), median(ρv)
        S40, I40 = glct_traj(rm, R0m, ipm, ρm, 40); S∞, _ = glct_traj(rm, R0m, ipm, ρm, 2000)
        say("$name at medians (r=$rm, R0=$(round(R0m,digits=3)), μ=$(round(ipm,digits=2)), ρ=$(round(ρm,digits=4))):")
        say("    N·I_40 = $(round(N*I40, digits=4)) people; I_40/I_peak-free: S_40=$(round(S40,digits=5)), S_∞=$(round(S∞,digits=5)); " *
            "N·(τ_∞ − τ_40) = $(round(N*(S40 - S∞), digits=4)) further infections")
        I40d = Float64[]; dτ = Float64[]
        for j in eachindex(rd)
            s40, i40 = glct_traj(rd[j], R0[j], ip[j], ρv[j], 40); s∞, _ = glct_traj(rd[j], R0[j], ip[j], ρv[j], 2000)
            push!(I40d, N * i40); push!(dτ, N * (s40 - s∞))
        end
        say("    over draws: N·I_40 = $(q(I40d)) people (99th pct $(round(quantile(I40d, 0.99), digits=3))); " *
            "N·(τ_∞ − τ_40) = $(q(dτ)) (99th pct $(round(quantile(dτ, 0.99), digits=3)))")
    end
    say("\n== Truncation check at T = 40 d (R0 ~ U(0.5, 5))")
    for (tag, name, obs, rec) in (("trunc_rung1", "Rung 1", :multbin, false),
                                  ("trunc_rung2", "Rung 2", :pois, false),
                                  ("trunc_rung6", "Rung 6", :multbin, true))
        m = ladder(:discrete, obs, rec); ch, t = fit(tag, m)
        trunc_report(name, m, ch)
    end

    # ---------------------------------------------------------- 3. Poisson column, two more seeds
    say("\n== Poisson-column cells, extra seeds (R0 ~ U(0.5, 5))")
    for seed in (2026, 31415), (name, obs) in (("Mult + Poisson(K;Nτ)", :multpois), ("Poisson counts      ", :pois))
        m = ladder(:discrete, obs, false); ch, t = fit("seed$(seed)_$(obs)", m; seed = seed)
        summ_ladder("seed $seed  $name", m, ch, t)
    end

    # ---------------------------------------------------------- 5b. fits with the never-infected cell
    say("\n== Count models with a never-infected cell (R0 ~ U(0.5, 5), and U(0.5, 20))")
    for hi in (5.0, 20.0)
        R0_HI[] = hi
        for (name, map) in (("discrete map + cell", :discrete), ("continuous ODE + cell", :ode))
            m = ladder(map, :multcell, false); ch, t = fit("cell_$(map)_R0hi$(Int(hi))", m)
            summ_ladder("R0≤$(Int(hi))  $name", m, ch, t)
        end
    end

elseif TASK == "orig"
    # ---------------------------------------------------------- 4. original count model, priors swapped
    # Data exactly as ddsa_vs_ode_count.qmd (ρ = 1/99); check it equals the ladder's data.
    Random.seed!(1234)
    Sh = glct_map(dph_nom.S, dph_nom.α, β_true, 1 / 99, nsteps)
    τ0 = Sh[1] - Sh[end]; f0 = max.(-diff(Sh), 0.0); f0 ./= sum(f0)
    K2 = rand(Binomial(N, τ0)); th2 = rand(Multinomial(K2, f0))
    say("Count-notebook data: K=$K2; identical onset histogram to the ladder data: $(th2 == th)")
    incidence = th2

    NegBin2nb(m, ϕ) = NegativeBinomial(ϕ, clamp(ϕ / (ϕ + m), 1e-6, 1.0 - 1e-6))   # as in the notebook
    function sir_exp_ode!(du, u, p, t)          # verbatim from ddsa_vs_ode_count.qmd
        (S, I, R, C) = u; (β, γ) = p; Npop = S + I + R
        inf = β * I / Npop * S
        du[1] = -inf; du[2] = inf - γ*I; du[3] = γ*I; du[4] = inf
    end
    function count_ll(y, Npop, i0, β, γ, ϕ)
        l = length(y); I0 = i0 * Npop
        prob = ODEProblem(sir_exp_ode!, [Npop - I0, I0, 0.0, 0.0], (0.0, float(l)), [β, γ])
        sol  = solve(prob, Tsit5(), saveat = 1.0)
        C    = Array(sol)[4, :]; newI = clamp.(diff(C), 1e-6, 1e7)
        (length(newI) == l && all(isfinite, newI)) || return -Inf
        return sum(logpdf(NegBin2nb(newI[i], ϕ), y[i]) for i in 1:l)
    end
    @model function count_exp_orig(y, Npop)       # model 2 of section 3(b), original priors
        i0 ~ Uniform(1e-4, 0.2); β ~ Uniform(1e-4, 2.5)
        γ  ~ Uniform(0.02, 1.0); ϕ ~ Exponential(5.0)
        Turing.@addlogprob! count_ll(y, Npop, i0, β, γ, ϕ)
    end
    @model function count_exp_R0mu(y, Npop, hi)   # (β, γ) prior → (R0, μ) prior; i0 prior kept
        i0 ~ Uniform(1e-4, 0.2); R0 ~ Uniform(0.5, hi)
        mean_ip ~ Uniform(1.0, 30.0); ϕ ~ Exponential(5.0)
        Turing.@addlogprob! count_ll(y, Npop, i0, R0 / mean_ip, 1 / mean_ip, ϕ)
    end
    @model function count_exp_ladder(y, Npop, hi) # all priors as the ladder
        ρ ~ Beta(1.0, Npop - 1); R0 ~ Uniform(0.5, hi)
        mean_ip ~ Uniform(1.0, 30.0); ϕ ~ Exponential(5.0)
        Turing.@addlogprob! count_ll(y, Npop, ρ, R0 / mean_ip, 1 / mean_ip, ϕ)
    end
    nuts_count = NUTS(2000, 0.95; max_depth = 12, Δ_max = 1000.0)   # notebook's count-model sampler
    function summ_orig(name, ch, t)
        if :R0 in names(ch)
            R0 = vec(ch[:R0]); ip = vec(ch[:mean_ip])
        else
            R0 = vec(ch[:β]) ./ vec(ch[:γ]); ip = 1 ./ vec(ch[:γ])
        end
        say("$name | R0=$(q(R0)) | meanIP=$(q(ip)) d | P(R0>5)=$(round(mean(R0 .> 5), digits=3)) | $(diagstr(ch)) | $(round(Int, t)) s")
    end
    for (tag, name, model) in (
            ("orig_priors",    "Original priors (β~U(1e-4,2.5), γ~U(0.02,1), i0~U(1e-4,0.2))", count_exp_orig(incidence, N)),
            ("orig_R0mu5",     "R0~U(0.5,5), μ~U(1,30); i0 prior kept                    ", count_exp_R0mu(incidence, N, 5.0)),
            ("orig_ladder5",   "Ladder priors: R0~U(0.5,5), μ~U(1,30), ρ~Beta(1,98)       ", count_exp_ladder(incidence, N, 5.0)),
            ("orig_ladder20",  "Ladder priors with R0~U(0.5,20)                            ", count_exp_ladder(incidence, N, 20.0)))
        ch, t = fit(tag, model; sampler = nuts_count, n_draws = 2000)
        summ_orig(name, ch, t)
    end
end
close(OUT)
