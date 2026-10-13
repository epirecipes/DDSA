# Supplement S2: ladder rungs 1–6 (+5b, 5c), the 2×2 that isolates the escapee term, and rung 1 with N unknown.
# Usage (from the supplementary folder): julia --project=@. -t 4 S2_identifiability/s2_1_rungs_2x2_unknownN.jl
#   RUNGS=1,2,…   subset of rungs (default all);  GRID2X2=1  the 2×2;  NWIDE=<N_max>  rung 1 with N ~ U(K, N_max)
# Outputs: out/s2_1_rungs.txt, s2_1_2x2.txt, s2_1_unknownN.txt (appended)
include(joinpath(@__DIR__, "s2_model.jl"))

# ---------------------------------------------------------------- fit
rungs = [
    ("1 DDSA Scenario B (discrete, Mult+Binom)", ladder(:discrete, :multbin, false)),
    ("2 Discrete map, Poisson counts",           ladder(:discrete, :pois,    false)),
    ("3 Discrete map, NegBin counts",            ladder(:discrete, :nb,      false)),
    ("4 Continuous ODE, Poisson counts",         ladder(:ode,      :pois,    false)),
    ("5 Continuous ODE, NegBin counts",          ladder(:ode,      :nb,      false)),
    ("6 DDSA full line list (Scenario A)",       ladder(:discrete, :multbin, true)),
    ("5b ODE NegBin, exponential IP",            ode_exp_same_priors()),
    ("5c ODE NegBin, exponential, count priors", ode_exp_count_priors()),
]
only = get(ENV, "RUNGS", "")
grid = get(ENV, "GRID2X2", "") == "1"
if grid
    rungs = [
        ("2x2 Mult + Binomial          (=rung 1)", ladder(:discrete, :multbin,   false)),
        ("2x2 Mult + Poisson(K;Nτ)     (≡rung 2)", ladder(:discrete, :multpois,  false)),
        ("2x2 Poisson counts + Binomial        ", ladder(:discrete, :poisbinom, false)),
        ("2x2 Poisson counts           (=rung 2)", ladder(:discrete, :pois,      false)),
    ]
    out0 = open(joinpath(RES, "s2_1_2x2.txt"), "w")
    for (r, R0, μ, ρ) in ((1, 3.1, 8.0, 0.01), (4, 2.5, 5.0, 0.02), (7, 4.6, 12.0, 0.005))
        a = ll_rung(r, R0, μ, ρ, 0.0; map=:discrete, obs=:multpois, recov=false)
        b = ll_rung(r, R0, μ, ρ, 0.0; map=:discrete, obs=:pois,     recov=false)
        println(out0, "identity check r=$r: Mult+Pois(K) = $a, Poisson counts = $b, diff = $(a-b)")
    end
    close(out0)
end
nwide = get(ENV, "NWIDE", "")
if !isempty(nwide)
    Nmax = parse(Int, nwide)
    N_GRID[] = NGrid(K, K, Nmax)
    rungs = [("1N Rung 1, N ~ DiscreteUniform($K, $Nmax)", ladder(:discrete, :multN, false))]
end
sel = only == "none" ? Int[] : isempty(only) ? eachindex(rungs) : parse.(Int, split(only, ","))

sampler = NUTS(1000, 0.8; max_depth = 12, Δ_max = 1000.0)
n_draws, n_chains = 4000, 4
out = open(joinpath(RES, !isempty(nwide) ? "s2_1_unknownN.txt" : grid ? "s2_1_2x2.txt" : "s2_1_rungs.txt"), "a")
for i in sel
    name, model = rungs[i]
    Random.seed!(1234)
    t = @elapsed ch = sample(model, sampler, MCMCThreads(), n_draws, n_chains; progress = false,
                             chain_type = MCMCChains.Chains, initial_params = fill(InitFromPrior(), n_chains))
    if :R0 in names(ch)
        R0 = vec(ch[:R0]); ip = vec(ch[:mean_ip])
    else
        R0 = vec(ch[:β]) ./ vec(ch[:γ]); ip = 1 ./ vec(ch[:γ])
    end
    pars = filter(p -> p in (:R0, :mean_ip, :ρ, :ϕ_, :β, :γ, :i0), names(ch))
    s = summarystats(ch[pars])
    rh = maximum(s[:, :rhat]); ess = minimum(s[:, :ess_bulk])
    div = sum(vec(ch[:numerical_error]))
    Pr = ""
    if i <= 6
        rd = [sample_index(l) for l in returned(model, ch)]
        Pr = "  P(r)=" * string(round.([mean(rd .== r) for r in 1:r_max], digits = 2))
        R0m, ipm, ρm = Array(ch[:R0]), Array(ch[:mean_ip]), Array(ch[:ρ])
        τd = [sum(mu_discrete(rd[j], R0m[j], ipm[j], ρm[j])[1]) / N for j in eachindex(rd)]
        Pr *= "  τ=" * q(vec(τd))
        if !isempty(nwide)
            Nd = [sample_N(N_GRID[], clamp(t, 0.001, 0.999)) for t in τd]
            Pr *= "  N=" * q(Nd) * "  P(N≤99)=$(round(mean(Nd .<= 99), digits=3))"
            nc = size(R0m, 2); Ndm = reshape(Nd, :, nc); τdm = reshape(τd, :, nc)
            for c in 1:nc
                Pr *= "\n    chain $c: R0=$(q(R0m[:,c]))  IP=$(q(ipm[:,c]))  τ=$(q(τdm[:,c]))  N=$(q(Ndm[:,c]))  ρ=$(q(ρm[:,c]))  ρN=$(q(ρm[:,c] .* Ndm[:,c]))"
            end
            lp = Array(ch[:logjoint]); Pr *= "\n    chain mean log-joint: $(round.(vec(mean(lp, dims=1)), digits=1))"
        end
    end
    ϕs = :ϕ_ in names(ch) ? "  ϕ=" * q(vec(ch[:ϕ_])) : ""
    line = "$name | R0=$(q(R0)) | meanIP=$(q(ip)) d | ρ=$(q(:ρ in names(ch) ? vec(ch[:ρ]) : vec(ch[:i0])))$ϕs$Pr | Rhat≤$(round(rh,digits=3)) ESS≥$(round(Int,ess)) div=$(Int(div)) | $(round(t,digits=0)) s"
    println(line); println(out, line); flush(out)
end
close(out)
