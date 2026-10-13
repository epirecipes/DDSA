# Supplement S2 (Table S8) and S8: attack-rate sweep by MCMC (rungs 1 and 2; R0 ≈ 1.5, 2, 3.16; N = 99, 1000). The chains feed the
# exact-grid comparison in s2_5_tau_sweep_grid.jl (Figure S2). CELLS=1.5,2.0,3.16 selects true R0 values. SWEEP_N, SWEEP_CHAINS, SWEEP_WARMUP, SWEEP_SUFFIX set N, chains, warm-up, output suffix;
# the N = 1000 comparison uses SWEEP_N=1000 SWEEP_CHAINS=12 SWEEP_WARMUP=2000 SWEEP_SUFFIX=_N1000_12ch.
# τ sweep: rungs 1 (Mult + Binomial) and 2 (Poisson counts) at R0 ≈ 1.5, 2, 3.16 and N = 99, 1000.
# Fisher information on τ: Binomial / Poisson = 1/(1−τ), so the Gaussian prediction of the SD ratio is sqrt(1/(1−τ)).
include(joinpath(@__DIR__, "s2_model.jl"))
using Serialization

const NS      = parse.(Int, split(get(ENV, "SWEEP_N", "99,1000"), ","))
const NCH     = parse(Int, get(ENV, "SWEEP_CHAINS", "4"))
const NWARM   = parse(Int, get(ENV, "SWEEP_WARMUP", "1000"))
const SUFFIX  = get(ENV, "SWEEP_SUFFIX", "")
const OUT = open(joinpath(RES, "s2_4_tausweep_mcmc$(SUFFIX)$(haskey(ENV, "CELLS") ? "_R0" * replace(ENV["CELLS"], "," => "_") : "").txt"), "w")
say(x) = (println(x); println(OUT, x); flush(OUT))
R0_HI[] = 20.0     # wide R0 prior so the Poisson rung is not cut at 5
say("R0 ~ U(0.5, 20), mean IP ~ U(1, 30), ρ ~ Beta(1, N−1), r summed 1..10; truth: Erlang-4 DPH (E[W] = $(round(EW, digits=3)) d), one seed (ρ = 1/N)")

"""Truth trajectory: S and I fractions after T steps."""
function truth_SI(β, ρ, T)
    S = 1 - ρ; x = dph_nom.α .* ρ
    for _ in 1:T
        ni = (1 - exp(-β * sum(x))) * S
        x = dph_nom.S' * x + dph_nom.α .* ni; S -= ni
    end
    return S, sum(x)
end

say("chains = $NCH, warm-up = $NWARM, draws = 2000 per chain")
for N_s in NS, R0_t in (1.5, 2.0, R0_imp)
    haskey(ENV, "CELLS") && !any(abs.(parse.(Float64, split(ENV["CELLS"], ",")) .- round(R0_t, digits=2)) .< 0.005) && continue
    β_s = R0_t / EW; ρ_s = 1 / N_s
    S∞, _ = truth_SI(β_s, ρ_s, 5000)
    # horizon: first T ≥ 40 with < 0.05 people infectious and < 0.05 infections still to come
    T = 40
    while true
        S_T, I_T = truth_SI(β_s, ρ_s, T)
        (N_s * I_T < 0.05 && N_s * (S_T - S∞) < 0.05) && break
        T += 5
    end
    # data: same generator as the ladder (deterministic map, K ~ Binomial, onsets ~ Multinomial)
    Random.seed!(1234)
    Sh = glct_map(dph_nom.S, dph_nom.α, β_s, ρ_s, T)
    τ_t = Sh[1] - Sh[end]; f_t = max.(-diff(Sh), 0.0); f_t ./= sum(f_t)
    K_s = rand(Binomial(N_s, τ_t)); th_s = rand(Multinomial(K_s, f_t))
    global N = N_s; global K = K_s; global th = th_s; global nsteps = T
    say("\n== N=$N_s, R0_true=$(round(R0_t, digits=2)), τ_true=$(round(τ_t, digits=3)), T=$T d, K=$K_s (K/N=$(round(K_s/N_s, digits=3))), " *
        "predicted info ratio 1/(1−τ)=$(round(1/(1-τ_t), digits=2)), SD ratio sqrt=$(round(sqrt(1/(1-τ_t)), digits=2))")
    res = Dict{Symbol,Any}()
    for (rung, obs) in ((1, :multbin), (2, :pois))
        m = ladder(:discrete, obs, false)
        Random.seed!(1234)
        t = @elapsed ch = sample(m, NUTS(NWARM, 0.8; max_depth = 12, Δ_max = 1000.0), MCMCThreads(), 2000, NCH;
                                 progress = false, chain_type = MCMCChains.Chains, initial_params = fill(InitFromPrior(), NCH))
        serialize(joinpath(CHAINS, "tau_N$(N_s)_R0$(round(R0_t, digits=2))_rung$rung$(SUFFIX).jls"), ch)
        rd = [sample_index(l) for l in returned(m, ch)]
        R0 = vec(ch[:R0]); ip = vec(ch[:mean_ip]); ρv = vec(ch[:ρ])
        τd = [sum(mu_discrete(rd[j], R0[j], ip[j], ρv[j])[1]) / N for j in eachindex(rd)]
        s = summarystats(ch[[:R0, :mean_ip, :ρ]])
        lo, hi = quantile(R0, [0.025, 0.975])
        res[Symbol("w$rung")] = hi - lo; res[Symbol("sd$rung")] = std(τd)
        res[Symbol("lw$rung")] = log(hi) - log(lo)
        say("  rung $rung: R0=$(q(R0)) width=$(round(hi-lo, digits=2)) | meanIP=$(q(ip)) | τ=$(q(τd)) SD(τ)=$(round(std(τd), digits=4)) | " *
            "P(R0>5)=$(round(mean(R0 .> 5), digits=3)) | R̂≤$(round(maximum(s[:, :rhat]), digits=3)) ESS≥$(round(Int, minimum(s[:, :ess_bulk]))) div=$(Int(sum(vec(ch[:numerical_error])))) | $(round(Int, t)) s")
        say("      R̂ by parameter: R0 $(round(s[:R0, :rhat], digits=3)) (ESS $(round(Int, s[:R0, :ess_bulk]))), meanIP $(round(s[:mean_ip, :rhat], digits=3)) (ESS $(round(Int, s[:mean_ip, :ess_bulk]))), ρ $(round(s[:ρ, :rhat], digits=3))")
        say("      per-chain median R0 $(round.(vec(median(Array(ch[:R0]), dims=1)), digits=2))")
        say("      per-chain median meanIP $(round.(vec(median(Array(ch[:mean_ip]), dims=1)), digits=2))")
        say("      per-chain mean log-joint $(round.(vec(mean(Array(ch[:logjoint]), dims=1)), digits=1))")
    end
    say("  ratios (rung 2 / rung 1): SD(τ) $(round(res[:sd2]/res[:sd1], digits=2)), R0 width $(round(res[:w2]/res[:w1], digits=2)), " *
        "log-R0 width $(round(res[:lw2]/res[:lw1], digits=2))  vs sqrt(1/(1−τ)) = $(round(sqrt(1/(1-τ_t)), digits=2))")
end
close(OUT)
