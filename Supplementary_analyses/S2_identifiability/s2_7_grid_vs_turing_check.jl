# Supplement S8 check: the grid posteriors of s2_5_tau_sweep_grid.jl evaluate the same log-joint as the Turing model
# `ladder` used for the MCMC fits. At random parameter points, compares the grid's log density (log-sum over r of
# ll_rung with a uniform prior on r = 1..10, plus the log prior densities of R0, the mean IP and ρ) with
# Turing.logjoint of `ladder`, for both infection-only observation models (rung 1: Multinomial + Binomial; rung 2:
# Poisson counts), on the ladder data (N = 99) and on the attack-rate-sweep data (N = 99 and 1,000, R0 = 1.5, 2, 3.16).
# Output: out/logs/s2_7_grid_vs_turing_check.log
# Usage (from the supplementary folder): julia --project=@. S2_identifiability/s2_7_grid_vs_turing_check.jl
include(joinpath(@__DIR__, "s2_model.jl"))
using Printf, Dates
lse(v) = (m = maximum(v); isfinite(m) ? m + log(sum(exp.(v .- m))) : m)
const LOGF = open(joinpath(RES, "logs", "s2_7_grid_vs_turing_check.log"), "w")
say(x) = (println(x); println(LOGF, x); flush(LOGF))
R0_HI[] = 20.0

# data of the attack-rate sweep, as in s2_5_tau_sweep_grid.jl (truth_SI and set_data! copied verbatim)
function truth_SI(β, ρ, T)
    S = 1 - ρ; x = dph_nom.α .* ρ
    for _ in 1:T
        ni = (1 - exp(-β * sum(x))) * S
        x = dph_nom.S' * x + dph_nom.α .* ni; S -= ni
    end
    return S, sum(x)
end
function set_data!(N_s, R0_t)
    β_s = R0_t / EW; ρ_s = 1 / N_s
    S∞, _ = truth_SI(β_s, ρ_s, 5000)
    T = 40
    while true
        S_T, I_T = truth_SI(β_s, ρ_s, T)
        (N_s * I_T < 0.05 && N_s * (S_T - S∞) < 0.05) && break
        T += 5
    end
    Random.seed!(1234)
    Sh = glct_map(dph_nom.S, dph_nom.α, β_s, ρ_s, T)
    τ_t = Sh[1] - Sh[end]; f_t = max.(-diff(Sh), 0.0); f_t ./= sum(f_t)
    K_s = rand(Binomial(N_s, τ_t)); th_s = rand(Multinomial(K_s, f_t))
    global N = N_s; global K = K_s; global th = th_s; global nsteps = T
    return τ_t
end

"""Grid-form log density at (R0, μ, ρ): as in grid_posterior, before multiplying by the cell widths."""
function grid_logdensity(obs, R0, μ, ρ)
    lls = [ll_rung(r, R0, μ, ρ, 0.0; map = :discrete, obs = obs, recov = false) for r in 1:r_max]
    return lse(lls .- log(r_max)) + logpdf(Uniform(0.5, R0_HI[]), R0) + logpdf(Uniform(1.0, 30.0), μ) + logpdf(Beta(1.0, N - 1), ρ)
end

function check(label)
    rng = Xoshiro(2026); maxd = 0.0; n = 0
    for obs in (:multbin, :pois), _ in 1:25
        R0 = 0.6 + 19rand(rng); μ = 1.5 + 28rand(rng); ρ = 10^(-4 + 3rand(rng))
        g = grid_logdensity(obs, R0, μ, ρ)
        isfinite(g) || continue
        lj = Turing.logjoint(ladder(:discrete, obs, false), (R0 = R0, mean_ip = μ, ρ = ρ))
        maxd = max(maxd, abs(lj - g)); n += 1
    end
    say(@sprintf("%-28s %3d finite points: max |Turing log-joint − grid log density| = %.2e", label, n, maxd))
    return maxd
end

say("Grid vs Turing log-joint check ($(Dates.now()))")
worst = check("ladder data (N = 99)")
for Ns in (99, 1000), R0t in (1.5, 2.0, R0_imp)
    set_data!(Ns, R0t)
    global worst = max(worst, check("sweep N = $Ns, R0 = $(round(R0t, digits = 2))"))
end
say(@sprintf("overall max difference: %.2e (supplement S8 states within 2e-12)", worst))
close(LOGF)
