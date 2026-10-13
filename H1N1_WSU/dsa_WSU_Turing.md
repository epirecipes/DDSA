# Inference using dynamic survival analysis and Turing.jl: WSU Data
Simon Frost (@sdwfrost), Sandra Montes (@slmontes)
2025-08-15

## Introduction

In this example, we use a discrete-time SIR model and Turing.jl to infer
the susceptible population $N$, the transmission rate ($\beta$),
recovery-rate parameter ($\gamma$), and the initial infected proportion
($\rho$) from line-list data. This dataset originates from a study
conducted on the H1N1 outbreak at Washington State University (WSU) in
2009, recording the number of daily cases over a period of 105 days
(days 0–104). The dataset was obtained from [KhudaBukhsh et
al. (2019)](https://github.com/cbskust/SDS.Epidemic/tree/master). The
original raw data only included daily counts of new infections;
KhudaBukhsh et al. reconstructed individual infection times from those
counts and generated individual recovery times by sampling from an
exponential distribution with rate $\gamma$ fixed at $1/5.5$ per day,
the mean of prior infectious-period estimates. The resulting dataset
(WSU.csv) contains individual-level infection times (first column) and
recovery times (second column) for 2,276 cases, including 13 individuals
whose recovery occurred after the end of the observation window (day
104) and are therefore treated as right-censored.

## Libraries

``` julia
using Distributions
using Turing
using MCMCChains   # sample(...; chain_type=MCMCChains.Chains)
using StatsPlots
using Plots
using Statistics
using Random
using CSV
using DataFrames
using MCMCDiagnosticTools
include("../IntegerParameters.jl")  # shared helpers: sum integer parameters (N, r) out of the likelihood
using .IntegerParameters
```

Set random seed for reproducibility

``` julia
seed = 1234
Random.seed!(seed)
```

## Utility functions

``` julia
"""
Convert a rate to a probability over time t.
"""
@inline function rate_to_proportion(r, t=1.0)
    if r < 0 || isnan(r)
        return 0.0
    elseif isinf(r)
        return 1.0
    else
        result = 1 - exp(-r * t)
        return isnan(result) ? 0.0 : max(min(result, 1.0), 0.0)
    end
end

"""
Generate a vector of counts from a vector of times and the total number of timesteps.
"""
function time_counts(times::AbstractVector{<:Integer}, nsteps::Integer)
    counts = zeros(Int, nsteps)
    @inbounds for t in times
        1 ≤ t ≤ nsteps && (counts[t] += 1)
    end
    return counts
end

"""
Prepare real data from DataFrame for analysis.
"""
function prep_sds(df::DataFrame, Tmax)
    ti     = Float64[]
    delta  = Float64[]
    cens   = Bool[]
    for r in eachrow(df)
        t_inf, t_rec = r[1], r[2]
        t_inf ≥ Tmax && continue                # infection after cut-off - ignore
        
        # Calculate recovery interval
        recovery_interval = t_rec - t_inf
        
        # Only include valid recovery intervals 
        if recovery_interval > 0
            push!(ti, t_inf)
            if t_rec < Tmax                        # full recovery observed
                push!(delta, recovery_interval)
                push!(cens, false)
            else                                   # right-censored
                push!(delta, Tmax - t_inf)
                push!(cens, true)
            end
        end
    end
    return ti, delta, cens
end

"""
Load and prepare real epidemic data from WSU.csv file.
"""
function load_real_data(csv_file::String, Tmax::Float64)
    # Load the data
    df = CSV.read(csv_file, DataFrame)
    
    # Prepare the data 
    ti, delta, cens = prep_sds(df, Tmax)
    
    # Convert to appropriate format for analysis
    K = length(ti)  # Total number of infections
    
    # Convert infection times to discrete time steps 
    # Calendar-day bins: step k holds the infection times in [k − 1, k), i.e. day k − 1 (day 0 is step 1).
    # The recorded times are daily counts spread within each day, so floor recovers the daily counts.
    ti_discrete = clamp.(floor.(Int, ti) .+ 1, 1, 1_000_000)
    # Observation window T = 104 d, as in KhudaBukhsh et al.'s code (T.max = 104): daily counts for days 0–104, where
    # the cases of day d ≥ 1 lie in (d − 1, d] and the 11 day-0 cases at t = 0.
    nsteps = max(maximum(ti_discrete), 104)
    
    # Create histogram of infection times
    th = time_counts(ti_discrete, nsteps)
    
    # Convert recovery intervals to integers (rounding to nearest day)
    delta_discrete = max.(1, round.(Int, delta))
    @assert sum(th) == K "sum(th) must equal K, check infection day discretisation"
    
    return Dict(
        :K => K,
        :ti => ti_discrete,
        :delta => delta_discrete,
        :th => th,
        :nsteps => nsteps,
        :cens => cens,
        :ti_original => ti,
        :delta_original => delta
    )
end
```

## SIR model

The model is a discrete-time SIR model with geometric recovery times. In
this model we assume that the recovery time is the same for all
individuals and that the recovery time is independent of the infection
time.

``` julia
function sir_map!(du, u, p, t)
    (S, I, C, Y) = u
    (β, γ) = p
    infection = rate_to_proportion(β*I)*S     # New infections
    recovery = rate_to_proportion(γ)*I        # New recoveries
    @inbounds begin
        du[1] = S - infection                 # Update susceptibles
        du[2] = I + infection-recovery        # Update infected
        du[3] = C + infection                 # Update cumulative infections
        du[4] = infection                     # Store new infections
    end
    nothing
end

function solve_map(f, u0, nsteps, p)
    # Pre-allocate array 
    sol = similar(u0, length(u0), nsteps + 1)
    # Initialise the first column with the initial state
    sol[:, 1] = u0
    # Iterate over the time steps
    @inbounds for t in 2:nsteps+1
        u = @view sol[:, t-1] # Get the current state
        du = @view sol[:, t]  # Prepare the next state
        f(du, u, p, t)        # Call the function to update du
    end
    return sol
end

function simulate(β, γ, ρ, nsteps)
    tspan = (0, nsteps)
    u0 = [1-ρ, ρ, 0.0, 0.0] # Initial conditions: S, I, C, Y
    p = [β, γ]
    sol = solve_map(sir_map!, u0, nsteps, p)
    τ = sol[3,end]      # Final size (total infected)
    f = sol[4,2:end]    # New infections per time step
    f = abs.(f)         # Ensure non-negative values
    # Handle edge cases 
    if τ <= 0 || any(isnan.(f)) || any(isinf.(f))
        f = fill(one(eltype(f))/nsteps, nsteps)
    else
        f = f./τ         # Normalise frequencies
    end 
    return sol, τ, f
end
```

## Data loading and preparation

Load the WSU data and prepare the data for inference.

``` julia
Tmax = 104.0   # end of the observation window (days 0–104); recoveries after it are right-censored
real_data = load_real_data("WSU.csv", Tmax)
```

Plot of the infection time observations

``` julia
bar(0:(real_data[:nsteps] - 1), real_data[:th], title="Infection time observations", xlabel="Time step", ylabel="Observations", legend=false)
```

![](dsa_WSU_Turing_files/figure-commonmark/cell-7-output-1.svg)

    Total infections: 2276
    Time steps: 104
    Number of censored observations: 13
    Mean infection time: 26.64
    Mean recovery interval: 5.0
    Peak infections: 169 at time step 18

## Turing model

The Turing.jl probabilistic model used to infer the parameters from this
dataset employs the previously defined discrete-time SIR model.

The susceptible population $N$ is integer-valued and enters the
likelihood only through the Binomial final-size term
$K \sim \mathrm{Binomial}(N, \tau)$. Instead of sampling it, we sum this
term over its prior support $N \in \{K+1, \ldots, N_{\max}\}$
(`marginal_N_loglik`, from the shared `IntegerParameters.jl`). The
discrete-time model and its likelihood are unchanged; only the remaining
real-valued parameters $(\beta, \gamma, \rho)$ are sampled, with NUTS.
Posterior draws of $N$ are recovered afterwards from its exact
conditional distribution given $\tau$.

``` julia
@model function dsa_discrete_turing_multinomial_inferN(
                                   N_grid::NGrid,           # Prior support of N: DiscreteUniform(K + 1, N_max)
                                   nsteps::Int,             # Number of time steps
                                   K::Int,                  # Observed total infections
                                   th::Vector{Int},         # Histogram of infection times
                                   delta::Vector{Int},      # Observed recovery intervals
                                   cens::Vector{Bool})      # Censoring (true=right-censored)
    
    # Priors for model parameters (N ~ DiscreteUniform(K + 1, N_max) is summed out below)
    β ~ Uniform(0.01, 5.0)   # Transmission rate 
    γ ~ truncated(LogNormal(log(1/5.5), 0.3), 0.01, 1.0)   # Recovery rate 
    ρ ~ Beta(1.0, 10.0)      # Initial infected proportion: small seed (mean 0.09), allowing the substantial
                             # fraction implied by a mid-epidemic start
    
    # Simulate deterministic model 
    _, τ, f = simulate(β, γ, ρ, nsteps)
    
    # Ensure τ is valid for binomial distribution
    τ_valid = max(min(τ, 1.0), 0.0)
    
    # Ensure f is valid for multinomial distribution
    f_valid = max.(f, 1e-10)           # Add small positive value to avoid zeros
    f_valid = f_valid ./ sum(f_valid)   # Normalise
    
    # Likelihood for number of infections, summed over N:
    #   log Σ_N P(N) · Binomial(K; N, τ)
    Turing.@addlogprob! marginal_N_loglik(N_grid, τ_valid)
    
    # Likelihood for infection times 
    th ~ Multinomial(K, f_valid)
    
    # Likelihood for recovery times 
    ϵ = 1e-8
    pγ = rate_to_proportion(γ)
    pγ_valid = min(max(pγ, ϵ), 1 - ϵ)
    log_surv = log1p(-pγ_valid)
    # 1-indexed geometric, P(W = n) = pγ (1 − pγ)^(n−1) for n ≥ 1, matching the per-step
    # recovery probability of the SIR map (Distributions.Geometric itself starts at 0)
    G1 = LocationScale(1, 1, Geometric(pγ_valid))
    for i in eachindex(delta)
        if cens[i]
            Turing.@addlogprob!(delta[i] * log_surv)  # log P(W > delta[i]) = delta[i]·log(1 − pγ)
        else
            delta[i] ~ G1
        end
    end
    
    return τ_valid   # used to draw N after sampling
end
```

Model parameters:

``` julia
N_max = 18000   # ≈ WSU student body (18,234 served by the campus health service; Vaidya et al. 2015)

th = real_data[:th]         # Infection times
K = sum(th)                 # Total infections
delta = real_data[:delta]   # Recovery intervals
cens  = real_data[:cens]    # Censoring
nsteps = real_data[:nsteps] # Number of time steps

N_grid = NGrid(K, K + 1, N_max)   # prior support of N, summed out in the likelihood
```

## Running the inference

``` julia
model = dsa_discrete_turing_multinomial_inferN(N_grid, nsteps, K, th, delta, cens);

n_samples  = 2000
nuts_adapt = 1000
n_chains   = 4
sampler    = NUTS(nuts_adapt, 0.8; max_depth=12, Δ_max=1000.0)
# Chains start from jittered data-based values (Methods): γ from the mean recorded recovery interval
# (p_γ = 1/mean δ), β at R0 = 1 (N is unknown, so the final size cannot be used) and ρ at its prior median;
# each value is multiplied by exp(0.1 z), z ~ N(0, 1), per chain.
init_rng = Xoshiro(20261006)
function data_inits(n)
    pγ0 = 1 / mean(delta[.!cens]); γ0 = -log(1 - pγ0); ρh = median(Beta(1.0, 10.0))
    j() = exp(0.1 * randn(init_rng))
    [InitFromParams((β = j() * pγ0, γ = clamp(j() * γ0, 0.011, 0.99), ρ = j() * ρh)) for _ in 1:n]
end

_ = sample(model, sampler, 1; progress=false, chain_type=MCMCChains.Chains)

@time chain = sample(model, sampler, MCMCThreads(), n_samples, n_chains;
                     progress=false, chain_type=MCMCChains.Chains, initial_params=data_inits(n_chains));

# Posterior draws of N: one draw from P(N | K, τ) per MCMC draw, added to the chain as :N
τ_draws = returned(model, chain)
N_draws = [sample_N(N_grid, τ) for τ in τ_draws]
chain   = hcat(chain, setrange(Chains(reshape(N_draws, size(N_draws, 1), 1, :), [:N]), range(chain)));
```

## Results

Once we have run the sampler, we can get a summary of the parameter
estimates

``` julia
describe(chain)
```

    Chains MCMC chain (2000×18×4 Array{Float64, 3}):

    Iterations        = 1001:1:3000
    Number of chains  = 4
    Samples per chain = 2000
    Wall duration     = 10.75 seconds
    Compute duration  = 39.51 seconds
    parameters        = β, γ, ρ, N
    internals         = n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size, logprior, loglikelihood, logjoint

    Summary Statistics

      parameters         mean         std      mcse    ess_bulk    ess_tail      r ⋯
          Symbol      Float64     Float64   Float64     Float64     Float64   Floa ⋯

               β       0.1902      0.0058    0.0001   2974.6294   3093.1007    1.0 ⋯
               γ       0.2172      0.0045    0.0001   3313.9705   3577.7443    1.0 ⋯
               ρ       0.0259      0.0028    0.0000   3264.8749   3349.6148    1.0 ⋯
               N   14176.5841   2017.1958   35.6290   3223.7847   3911.0705    1.0 ⋯

                                                                   2 columns omitted

    Quantiles

      parameters         2.5%        25.0%        50.0%        75.0%        97.5% 
          Symbol      Float64      Float64      Float64      Float64      Float64 

               β       0.1795       0.1862       0.1900       0.1939       0.2027
               γ       0.2083       0.2142       0.2172       0.2203       0.2261
               ρ       0.0212       0.0237       0.0257       0.0278       0.0318
               N   10358.8750   12656.0000   14144.0000   15776.0000   17727.0000

Plot trace plots for MCMC diagnostics

``` julia
plot(chain, plot_title="Trace plots")
```

![](dsa_WSU_Turing_files/figure-commonmark/cell-13-output-1.svg)

Extract parameter estimates median values

``` julia
# Extract parameter estimates
N_post = round(Int, median(chain[:N]))
β_post = median(chain[:β])
γ_post = median(chain[:γ])
ρ_post = median(chain[:ρ])
```

Posterior summaries with 95% credible intervals. Derived quantities (R₀
and mean infectious period) are computed sample-by-sample from the
chain.

``` julia
N_chain  = vec(chain[:N])
β_chain  = vec(chain[:β])
γ_chain  = vec(chain[:γ])
ρ_chain  = vec(chain[:ρ])
pγ_chain = rate_to_proportion.(γ_chain)

R0_chain       = β_chain ./ pγ_chain   # R0 of the discrete process = β / (1 - exp(-γ))
IP_steps_chain = 1 ./ pγ_chain         # mean infectious period in steps (geometric, the actual fitted process)
tau_chain      = K ./ N_chain          # observed attack rate posterior

function posterior_summary(x; digits=4)
    (median = round(median(x); digits=digits),
     mean   = round(mean(x);   digits=digits),
     q025   = round(quantile(x, 0.025); digits=digits),
     q975   = round(quantile(x, 0.975); digits=digits))
end

posterior_table = (
    N         = posterior_summary(N_chain; digits=0),
    β         = posterior_summary(β_chain),
    γ         = posterior_summary(γ_chain),
    ρ         = posterior_summary(ρ_chain),
    R0        = posterior_summary(R0_chain),
    IP_steps  = posterior_summary(IP_steps_chain; digits=2),
    K_over_N  = posterior_summary(tau_chain),
)
```

    (N = (median = 14144.0, mean = 14177.0, q025 = 10359.0, q975 = 17727.0), β = (median = 0.19, mean = 0.1902, q025 = 0.1795, q975 = 0.2027), γ = (median = 0.2172, mean = 0.2172, q025 = 0.2083, q975 = 0.2261), ρ = (median = 0.0257, mean = 0.0259, q025 = 0.0212, q975 = 0.0318), R0 = (median = 0.9714, mean = 0.9745, q025 = 0.9394, q975 = 1.028), IP_steps = (median = 5.12, mean = 5.12, q025 = 4.94, q975 = 5.32), K_over_N = (median = 0.1609, mean = 0.164, q025 = 0.1284, q975 = 0.2197))

    N           median = 14144.0   mean = 14177.0   95% CI = [10359.0, 17727.0]
    β           median = 0.19   mean = 0.1902   95% CI = [0.1795, 0.2027]
    γ           median = 0.2172   mean = 0.2172   95% CI = [0.2083, 0.2261]
    ρ           median = 0.0257   mean = 0.0259   95% CI = [0.0212, 0.0318]
    R0          median = 0.9714   mean = 0.9745   95% CI = [0.9394, 1.028]
    IP_steps    median = 5.12   mean = 5.12   95% CI = [4.94, 5.32]
    K_over_N    median = 0.1609   mean = 0.164   95% CI = [0.1284, 0.2197]

    90% credible intervals:
    N           median = 14144.0   90% CI = [10898.0, 17453.0]
    β           median = 0.19   90% CI = [0.1812, 0.2003]
    γ           median = 0.2172   90% CI = [0.2097, 0.2247]
    ρ           median = 0.0257   90% CI = [0.0217, 0.0309]
    R0          median = 0.971   90% CI = [0.943, 1.017]
    IP_steps    median = 5.12   90% CI = [4.97, 5.29]

    Convergence diagnostics:

    ESS

      parameters         ess   ess_per_sec 
          Symbol     Float64       Float64 

               β   2974.6294       75.2899
               γ   3313.9705       83.8789
               ρ   3264.8749       82.6362
               N   3223.7847       81.5962

    R-hat

      parameters      rhat 
          Symbol   Float64 

               β    1.0023
               γ    1.0022
               ρ    1.0012
               N    1.0012

Plot model fit

![](dsa_WSU_Turing_files/figure-commonmark/cell-17-output-1.svg)

![](dsa_WSU_Turing_files/figure-commonmark/cell-18-output-1.svg)

    Final epidemic size (simulated, at median N): τ * N̂ = 2293.9
    Total observed infections: 2276
    Inferred susceptible population (median): N̂ = 14144
    Mean recovery time (observed delta, hosp - onset): 5.0 days
    Mean recovery delta (fitted, 1/p_γ̂): 5.12 days
    Observed total infections: 2276
    Predicted total infections: 2293.9
    Recovery delta fit error (observed vs 1/p_γ̂): 0.9%

## Comparison to KhudaBukhsh et al. (2019) results

We plot the S_t curve and the final epidemic size distribution to
compare our observed and simulated data, with that obtained in
KhudaBukhsh et al. (2019).

![](dsa_WSU_Turing_files/figure-commonmark/cell-20-output-1.svg)

![](dsa_WSU_Turing_files/figure-commonmark/cell-21-output-1.svg)

    Final size, posterior predictive replicates Binomial(N, τ) (observed K = 2276): 
    median = 2276.0   
    95% interval = [2156.0, 2397.0]

## Manuscript figure export

    Saved -> ../figures/fig_h1n1.pdf

<div id="fig-h1n1">

    Plots.GRBackend()

Figure 1

</div>

## Discussion

The analysis performed on the WSU H1N1 outbreak dataset from 2009 showed
that the discrete SIR model, with geometric recovery times, follows the
overall decline of incidence but not the two early peaks (days 10 and
17). The population at risk $N$ has a posterior median of $14{,}144$
(95% CI $[10{,}359,\,17{,}727]$), larger than the estimate of $7{,}051$
(90% CI $[6{,}602,\,7{,}581]$) reported by KhudaBukhsh et al. (2020).
The two analyses treat $N$ differently: here it enters through the
Binomial final-size term and is summed out over a uniform prior, whereas
KhudaBukhsh et al. impute it at each iteration as
$n \sim \mathrm{NegBinom}(k, \tau)$ from the final-size equation and fix
$\gamma^{-1} = 5.5$ days. The transmission rate ($\beta = 0.190$) agrees
with theirs ($0.189$). The reproduction number ($R_0 = 0.97$, 95% CI
$[0.94,\,1.03]$) is slightly lower than theirs ($1.06$), which is
consistent with the larger $N$, since it implies a smaller attack rate.
Both estimates of $R_0$ lie close to 1. The fitted mean infectious
period ($5.12$ days) reproduces the mean recorded recovery interval.

This approach offers several advantages: a simple and interpretable
modelling framework, fast computation and inference through Turing.jl,
and clear parameter interpretation (transmission rate β, recovery rate
γ, and initial infected proportion ρ). However, it’s important to note
that the model relies on key assumptions, including geometric recovery
times and the modification of the original dataset to include recovery
intervals, as the primary data only contained daily infection counts.
