# Supplement S2: data, stage supports and trajectory functions shared by s2_8_grid_poisson.jl (numerical grid posteriors of
# the models without recovery intervals), s2_9_trajectories.jl (full-data DDSA and the trajectory figures) and
# s2_10_trajectory_panels.jl (the trajectories of all four models at the three population sizes in one figure).
# Include after s2_model.jl, whose globals hold the N = 99 ladder data when this file is included.
# Data. N = 99: the ladder data of s2_model.jl. N = 1,000 and 10,000: the attack-rate sweep generator of s2_4_tau_sweep_mcmc.jl
# at R0 = 3.16 (at N = 1,000 the same infection times as that sweep cell), plus recovery intervals drawn from the simulator's
# DPH as in s2_model.jl. N is the whole population, including the initial infectives (supplement S2 notation); the generator
# starts the map at S_0 = 1 − ρ, I_0 = ρ (fractions of N) and draws K ~ Binomial(N, τ). ρ = 0.01 at N = 99 (ρN = 0.99, as in
# the ladder data) and 1/N otherwise; the line list holds one recovery interval for the initial infective and one for each case.
# Stages. The discrete map sums r over 1..r_max (= 10) with r < μ (p = r/μ < 1). The ODE sums r over 1..R_ODE with no
# constraint on μ (r/μ is its stage rate): the ladder's μ > r, which only the map needs, would let the ODE have a
# low-variability infectious period only at a long mean (the data's has CV 0.30, which an Erlang-r matches at r ≈ 11).
const DATA99 = (N = N, K = K, th = copy(th), delta = copy(delta), nsteps = nsteps, ρ = ρ_true, β = β_true)
const R_ODE = 40                         # Erlang stages of the ODE (at N = 10,000 its posterior of r rises to the top of 1..20)
rsupport(m) = m === :ode ? (1:R_ODE) : (1:r_max)   # m = :ode, or the discrete map (:map, :discrete)

function truth_SI(β, ρ, T)
    S = 1 - ρ; x = dph_nom.α .* ρ
    for _ in 1:T
        ni = (1 - exp(-β * sum(x))) * S
        x = dph_nom.S' * x + dph_nom.α .* ni; S -= ni
    end
    return S, sum(x)
end
"""Set the globals N, K, th, delta, nsteps used by the likelihoods; return the true (β, ρ)."""
function set_data!(N_s)
    if N_s == 99
        global N = DATA99.N; global K = DATA99.K; global th = DATA99.th; global delta = DATA99.delta; global nsteps = DATA99.nsteps
        return DATA99.β, DATA99.ρ
    end
    β_s = R0_imp / EW; ρ_s = 1 / N_s
    S∞, _ = truth_SI(β_s, ρ_s, 5000)
    T = 40                               # first T ≥ 40 with < 0.05 people infectious and < 0.05 infections still to come
    while true
        S_T, I_T = truth_SI(β_s, ρ_s, T)
        (N_s * I_T < 0.05 && N_s * (S_T - S∞) < 0.05) && break
        T += 5
    end
    Random.seed!(1234)
    Sh = glct_map(dph_nom.S, dph_nom.α, β_s, ρ_s, T)
    τ_t = Sh[1] - Sh[end]; f_t = max.(-diff(Sh), 0.0); f_t ./= sum(f_t)
    K_s = rand(Binomial(N_s, τ_t)); th_s = rand(Multinomial(K_s, f_t))
    dph = PhaseTypeDistributions.DPH(dph_nom.α, dph_nom.S)
    δ = Int[]
    for _ in 1:(1 + K_s)                 # the cases and the initial infective
        u = rand(); cum = 0.0; rt = 1
        while cum < u
            cum += PhaseTypeDistributions.pmf(dph, rt); cum >= u && break; rt += 1
        end
        push!(δ, rt)
    end
    global N = N_s; global K = K_s; global th = th_s; global delta = δ; global nsteps = T
    return β_s, ρ_s
end

# ---------------------------------------------------------------- trajectories
"""Discrete map: expected new infections in each step (k = 1..T) and number infectious after each step (k = 0..T)."""
function glct_traj(Tm, α, β, ρ, T)
    S = 1 - ρ; x = α .* ρ
    inc = zeros(T); prev = zeros(T + 1); prev[1] = N * ρ
    for k in 1:T
        ni = (1 - exp(-β * sum(x))) * S
        x = Tm' * x + α .* ni; S -= ni
        inc[k] = N * ni; prev[k+1] = N * sum(x)
    end
    return inc, prev
end
function map_traj(r, R0, μ, ρ, T)
    dph = PhaseTypeDistributions.dph_negative_binomial(r, r / μ)
    return glct_traj(dph.S, dph.α, R0 / μ, ρ, T)
end
"""Erlang-r ODE (solver settings of mu_ode): new infections over (k − 1, k] and number infectious at t = 0..T."""
function ode_traj(r, R0, μ, ρ, T)
    u0 = zeros(r + 2); u0[1] = 1 - ρ; u0[2] = ρ
    sol = solve(ODEProblem(erlang_ode!, u0, (0.0, float(T)), (R0 / μ, r / μ, r)), Tsit5();
                saveat = 1.0, abstol = 1e-9, reltol = 1e-7)
    (SciMLBase.successful_retcode(sol) && length(sol.u) == T + 1) || return fill(NaN, T), fill(NaN, T + 1)
    return N .* max.(diff([s[r+2] for s in sol.u]), 0.0), N .* [sum(s[2:r+1]) for s in sol.u]
end
"""Number infectious in the line list after each step (k = 0..T): the initial infective from step 0, each case from its step."""
function linelist_prev(th, delta, T)
    start = vcat(0, reduce(vcat, [fill(k, th[k]) for k in eachindex(th)]))
    @assert length(start) == length(delta)
    return [count(i -> start[i] <= k <= start[i] + delta[i] - 1, eachindex(start)) for k in 0:T]
end
