# Supplement S2: likelihood checks (rung 1 against the data-availability notebook; ODE vs discrete; ODE gradient)
include(joinpath(@__DIR__, "s2_model.jl"))
# Rung 1 vs notebook's dph_infonly_loglik at test points; Poisson log-likelihood, discrete map vs ODE
function nb_infonly(nb_r, R₀, μ, ρ)
    dph = PhaseTypeDistributions.dph_negative_binomial(nb_r, nb_r/μ)
    Sh = glct_map(dph.S, dph.α, R₀/μ, ρ, nsteps); τ = Sh[1]-Sh[end]
    f = max.(-diff(Sh),0.0); f ./= sum(f)
    binomlogpdf(N, clamp(τ,0.001,0.999), K) + logpdf(Multinomial(K, f), th)
end
for r in (1,4,7)
    a = ll_rung(r, 3.1, 8.0, 0.01, 0.0; map=:discrete, obs=:multbin, recov=false)
    println("r=$r rung1=$a notebook=$(nb_infonly(r,3.1,8.0,0.01))")
    println("   pois discrete=$(ll_rung(r,3.1,8.0,0.01,0.0;map=:discrete,obs=:pois,recov=false))  ode=$(ll_rung(r,3.1,8.0,0.01,0.0;map=:ode,obs=:pois,recov=false))")
end
using ForwardDiff
g = ForwardDiff.gradient(x -> ll_rung(4, x[1], x[2], x[3], 2.0; map=:ode, obs=:nb, recov=false), [3.1, 8.0, 0.01])
println("ODE NB grad: $g")
@time for _ in 1:10; ForwardDiff.gradient(x -> ll_rung(4, x[1], x[2], x[3], 2.0; map=:ode, obs=:nb, recov=false), [3.1, 8.0, 0.01]); end
