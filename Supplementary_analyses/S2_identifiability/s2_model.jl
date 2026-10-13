# Supplement S2 (identifiability ladder): data, forward maps, likelihoods and the `ladder` model, shared by all s2_* scripts.
# Ladder: Scenario B -> count likelihoods -> continuous ODE -> full line list.
# Rung 2 changes the observation likelihood of rung 1; rungs 2–5 cross forward model (discrete map / Erlang-r ODE)
# with count likelihood (Poisson / NegBin); rung 6 adds ℓ_R to rung 1. Same data, same priors, N = 99.
using Turing, Distributions, Random, Statistics, LinearAlgebra
using MCMCChains, MCMCDiagnosticTools, OrdinaryDiffEq, StatsFuns
const ROOT = let d = @__DIR__   # repository root: nearest parent folder that holds the shared modules
    while !isfile(joinpath(d, "IntegerParameters.jl")) && dirname(d) != d; d = dirname(d); end; d
end
const SA     = normpath(joinpath(@__DIR__, ".."))              # the supplementary folder
const RES    = mkpath(joinpath(@__DIR__, "out"))
const FIGS   = mkpath(joinpath(@__DIR__, "figs"))
const CHAINS = mkpath(joinpath(@__DIR__, "chains"))
include(joinpath(ROOT, "PhaseTypeDistributions.jl"))
include(joinpath(ROOT, "IntegerParameters.jl")); using .IntegerParameters

# ---------------------------------------------------------------- data (as ddsa_data_avail.qmd)
Random.seed!(1234)
R0_cal, ρ_true, μ_ip, k, Δt = 2.0, 0.01, 4.0, 4, 1.0
nsteps = 40
β_true = R0_cal / μ_ip * Δt
dph_nom = PhaseTypeDistributions.dph_erlang_from_mean(k, μ_ip, Δt)
EW = PhaseTypeDistributions.mean(dph_nom); R0_imp = β_true * EW

function glct_map(Tm, α, β, ρ, nsteps)
    T = promote_type(eltype(Tm), typeof(β), typeof(ρ))
    S = one(T) - ρ; x = α .* ρ
    Sh = Vector{T}(undef, nsteps + 1); Sh[1] = S
    for j in 1:nsteps
        new_inf = (1 - exp(-β * sum(x))) * S
        x = Tm' * x + α .* new_inf
        S -= new_inf; Sh[j+1] = S
    end
    return Sh
end

N, M = 99, 1
let Sh = glct_map(dph_nom.S, dph_nom.α, β_true, ρ_true, nsteps)
    τ = Sh[1] - Sh[end]; f = max.(-diff(Sh), 0.0); f ./= sum(f)
    global K = rand(Binomial(N, τ)); global th = rand(Multinomial(K, f))
    dph = PhaseTypeDistributions.DPH(dph_nom.α, dph_nom.S)
    global delta = Int[]
    for _ in 1:(M + K)
        u = rand(); cum = 0.0; rt = 1
        while cum < u
            cum += PhaseTypeDistributions.pmf(dph, rt); cum >= u && break; rt += 1
        end
        push!(delta, rt)
    end
end
println("Data: K=$K/$N, recovery mean $(round(mean(delta),digits=2)) d, min $(minimum(delta))")
@assert K == 91 && minimum(delta) == 4   # same realisation as the data-availability notebook

# ---------------------------------------------------------------- forward models
# Expected daily new infections (counts) for phase count r, mean IP μ, R0, ρ.
function mu_discrete(r, R0, μ, ρ)
    p = r / μ
    dph = PhaseTypeDistributions.dph_negative_binomial(r, p)
    Sh = glct_map(dph.S, dph.α, R0 / μ, ρ, nsteps)
    return N .* max.(-diff(Sh), 0.0), dph
end

function erlang_ode!(du, u, p, t)          # u = [S, I_1..I_r, C], fractions
    β, λ, r = p
    Itot = zero(eltype(u)); for j in 1:r; Itot += u[1+j]; end
    inf = β * u[1] * Itot
    du[1] = -inf
    du[2] = inf - λ * u[2]
    for j in 2:r; du[1+j] = λ * (u[j] - u[1+j]); end
    du[r+2] = inf
end

function mu_ode(r, R0, μ, ρ)                # continuous Erlang-r SIR, stage rate r/μ
    T = promote_type(typeof(R0), typeof(μ), typeof(ρ))
    u0 = zeros(T, r + 2); u0[1] = 1 - ρ; u0[2] = ρ
    prob = ODEProblem(erlang_ode!, u0, (0.0, float(nsteps)), (R0 / μ, r / μ, r))
    sol = solve(prob, Tsit5(); saveat = 1.0, abstol = 1e-9, reltol = 1e-7)
    SciMLBase.successful_retcode(sol) || return nothing
    C = [s[r+2] for s in sol.u]
    length(C) == nsteps + 1 || return nothing
    return N .* max.(diff(C), zero(T))
end

function safe_normalise(f::AbstractVector{T}, n::Int) where T
    f_pos = ifelse.(isnan.(f) .| .!isfinite.(f), zero(T), max.(f, zero(T)))
    sF = sum(f_pos)
    (sF > T(1e-15) && isfinite(sF)) ? f_pos ./ sF : fill(one(T)/n, n)
end

const N_GRID = Ref(NGrid(91, 91, 99))

NegBin2(m, ϕ) = NegativeBinomial(ϕ, clamp(ϕ / (ϕ + m), 1e-10, 1 - 1e-10))

# ---------------------------------------------------------------- per-r log-likelihoods
# obs: :multbin (Multinomial + Binomial final size), :pois, :nb
function ll_counts(μt, obs, ϕ)
    any(!isfinite, μt) && return -Inf
    if obs === :multbin
        τ = sum(μt) / N
        τv = clamp(τ, 0.001, 0.999)
        f = safe_normalise(μt, nsteps)   # as in the notebooks: uniform fallback when τ ≈ 0
        (all(isfinite, f) && abs(sum(f) - 1) < 0.01) || return -Inf
        return binomlogpdf(N, τv, K) + logpdf(Multinomial(K, f), th)
    elseif obs === :multcell         # counts + never-infected cell: Multinomial(N; τf_1..τf_T, 1 − τ)
        f = safe_normalise(μt, nsteps)
        (all(isfinite, f) && abs(sum(f) - 1) < 0.01) || return -Inf
        τv = clamp(sum(μt) / N, 0.001, 0.999)
        return logpdf(Multinomial(N, vcat(τv .* f, 1 - τv)), vcat(th, N - K))
    elseif obs === :multN            # Multinomial + Binomial(N, τ) with N summed out over N_GRID
        f = safe_normalise(μt, nsteps)
        (all(isfinite, f) && abs(sum(f) - 1) < 0.01) || return -Inf
        τv = clamp(sum(μt) / N, 0.001, 0.999)   # τ is a fraction; N here only scales μt
        return marginal_N_loglik(N_GRID[], τv) + logpdf(Multinomial(K, f), th)
    elseif obs === :multpois         # Multinomial timing + Poisson(K; Nτ) final size
        f = safe_normalise(μt, nsteps)
        (all(isfinite, f) && abs(sum(f) - 1) < 0.01) || return -Inf
        return poislogpdf(max(sum(μt), 1e-10), K) + logpdf(Multinomial(K, f), th)
    elseif obs === :poisbinom        # Poisson daily counts + explicit Binomial(N, τ) final size
        m = max.(μt, 1e-10)
        τv = clamp(sum(μt) / N, 0.001, 0.999)
        return sum(poislogpdf(m[i], th[i]) for i in eachindex(th)) + binomlogpdf(N, τv, K)
    elseif obs === :pois
        m = max.(μt, 1e-10)
        return sum(poislogpdf(m[i], th[i]) for i in eachindex(th))
    else
        m = max.(μt, 1e-10)
        return sum(logpdf(NegBin2(m[i], ϕ), th[i]) for i in eachindex(th))
    end
end

function ll_rung(r, R0, μ, ρ, ϕ; map::Symbol, obs::Symbol, recov::Bool)
    r / μ < 1 || return -Inf               # same support in every rung (discrete map needs p < 1)
    if map === :discrete
        μt, dph = mu_discrete(r, R0, μ, ρ)
    else
        μt = mu_ode(r, R0, μ, ρ); μt === nothing && return -Inf
    end
    ll = ll_counts(μt, obs, ϕ)
    if recov
        any(<(r), delta) && return -Inf
        dph = PhaseTypeDistributions.dph_negative_binomial(r, r / μ)
        ll += sum(logpdf(dph, d) for d in delta)
    end
    return ll
end

const r_max = 10
const R0_HI = Ref(5.0)   # upper bound of the R0 prior
@model function ladder(map::Symbol, obs::Symbol, recov::Bool)
    R0 ~ Uniform(0.5, R0_HI[])
    mean_ip ~ Uniform(1.0, 30.0)
    ρ ~ Beta(1.0, N - 1)
    if obs === :nb
        ϕ_ ~ Exponential(5.0); ϕ = ϕ_
    else
        ϕ = 0.0
    end
    logmarg, lps = marginal_r_loglik(1:r_max) do r
        ll_rung(r, R0, mean_ip, ρ, ϕ; map, obs, recov)
    end
    Turing.@addlogprob! logmarg
    return lps
end

# Extra diagnostic rungs (continuous ODE + NegBin), not part of the one-factor ladder:
#  5b: same priors, exponential IP (r ≡ 1)
#  5c: exponential IP and the priors of ddsa_vs_ode_count.qmd (i0, β, γ uniform)
@model function ode_exp_same_priors()
    R0 ~ Uniform(0.5, R0_HI[]); mean_ip ~ Uniform(1.0, 30.0); ρ ~ Beta(1.0, N - 1); ϕ_ ~ Exponential(5.0)
    μt = mu_ode(1, R0, mean_ip, ρ)
    μt === nothing && (Turing.@addlogprob!(-Inf); return)
    Turing.@addlogprob! ll_counts(μt, :nb, ϕ_)
end
@model function ode_exp_count_priors()
    i0 ~ Uniform(1e-4, 0.2); β ~ Uniform(1e-4, 2.5); γ ~ Uniform(0.02, 1.0); ϕ_ ~ Exponential(5.0)
    μt = mu_ode(1, β / γ, 1 / γ, i0)
    μt === nothing && (Turing.@addlogprob!(-Inf); return)
    Turing.@addlogprob! ll_counts(μt, :nb, ϕ_)
end


q(v) = "$(round(median(v),digits=2)) [$(round(quantile(v,0.025),digits=2)), $(round(quantile(v,0.975),digits=2))]"
