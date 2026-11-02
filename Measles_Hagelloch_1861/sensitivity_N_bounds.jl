# Hagelloch sensitivity to the class-size priors (main text Section 3(e); not part of the notebook).
# The notebook bounds each class separately, N_i ∈ K_i : (197 − Σ_{j≠i} K_j), which jointly allows N0 + N1 + N2 up to
# 215, although the bounds come from 197 children under 14; the 3 teenage cases (aged 14–15, all in class 2) are not
# among the 197. This script refits the GLCT model under:
#   (b) joint   N_i ≥ K_i and N0 + N1 + N2 ≤ 197 + 3 = 200, uniform over the feasible triples (N2 ≤ 80, N0 ≤ 102, N1 ≤ 42)
#   (c) fixed   N_i = K_i (attack rate 100%, as Neal & Roberts 2004 assume, n = m = 188)
# and checks that the shared likelihood reproduces the notebook's (a) at the current bounds (constant offset only).
# The model code (data, maps, hagelloch_cell, priors, sampler, data-based starts) is taken verbatim from the notebook's
# julia chunks, Libraries through Sampler. Integer parameters are summed out exactly; posterior draws recovered after.
# Usage (repo root):  julia --project=. -t 10 Measles_Hagelloch_1861/sensitivity_N_bounds.jl      (VARIANTS=joint,fixed)
ENV["GKSwstype"] = "100"
const HDIR = @__DIR__
const IS_MAIN = abspath(PROGRAM_FILE) == @__FILE__   # before cd: PROGRAM_FILE may be relative
cd(HDIR)

# ---- notebook code, chunks from Libraries through Sampler (the histogram plot chunk is skipped)
let lines = readlines(joinpath(HDIR, "dsa_hagelloch_GLCTvsBoxcar.qmd")), chunks = String[], buf = String[], inside = false
    for l in lines
        if !inside && startswith(l, "```{julia}"); inside = true; empty!(buf)
        elseif inside && startswith(l, "```"); inside = false; push!(chunks, join(buf, "\n"))
        elseif inside; push!(buf, l); end
    end
    last = findfirst(c -> occursin("function add_integer_draws", c), chunks)
    code = join([c for c in chunks[1:last] if !occursin("bar(1:nsteps", c)], "\n\n")
    include_string(Main, code, joinpath(HDIR, "dsa_hagelloch_GLCTvsBoxcar.qmd.jl"))
end
using Printf
import Distributions.SpecialFunctions: loggamma

const OUTD = joinpath(HDIR, "sensitivity_out"); mkpath(OUTD)
const N_JOINT = 197 + 3

logbinom(N, K, τ) = loggamma(N + 1.0) - loggamma(K + 1.0) - loggamma(N - K + 1.0) + K * log(τ) + (N - K) * log1p(-τ)
lse(v) = (m = maximum(v); isfinite(m) ? m + log(sum(exp.(v .- m))) : m)

"""Allowed (N0, N1) pairs for each N2 under a variant; `cells` are the (nb_r, N2) combinations."""
function variant_support(v)
    if v === :current
        N2r = K2:N2_max; pairs = Dict(N2 => [(a, b) for a in K0:N0_max for b in K1:N1_max] for N2 in N2r)
    elseif v === :joint
        N2r = K2:(N_JOINT - K0 - K1)
        pairs = Dict(N2 => [(a, b) for a in K0:(N_JOINT - K1 - N2) for b in K1:(N_JOINT - a - N2)] for N2 in N2r)
    elseif v === :fixed
        N2r = K2:K2; pairs = Dict(K2 => [(K0, K1)])
    end
    cells = vec(collect(Iterators.product(1:nb_r_max, N2r)))
    return cells, pairs
end

"""hagelloch_cell with the N0, N1 Binomial terms summed (not averaged) over the allowed pairs, so that the uniform
prior over (nb_r, N2) cells used by marginal_r_loglik, times this sum, is a uniform prior over feasible triples."""
function cell_ll(propagate, nb_r, N2, mean_δ, pairs)
    nb_p = nb_r / (mean_δ - δ_floor_ip + nb_r)
    dph  = PhaseTypeDistributions.dph_negative_binomial_min_support(nb_r, nb_p, δ_floor_ip)
    sim  = propagate(dph, 1.0 / N2)
    tau0 = clamp(sim.tau0, 0.001, 0.999); tau1 = clamp(sim.tau1, 0.001, 0.999); tau2 = clamp(sim.tau2, 0.001, 0.999)
    f0 = safe_normalise(sim.f0, nsteps); f1 = safe_normalise(sim.f1, nsteps); f2 = safe_normalise(sim.f2, nsteps)
    terms = [logbinom(a, K0, tau0) + logbinom(b, K1, tau1) for (a, b) in pairs]
    ll = lse(terms) + logpdf(Binomial(N2, tau2), K2)
    for (K, f, th) in ((K0, f0, th0), (K1, f1, th1), (K2, f2, th2))
        (K > 0 && all(isfinite, f) && abs(sum(f) - 1) < 0.01) || return (ll = -Inf, terms = terms)
        ll += logpdf(Multinomial(K, f), th)
    end
    ll += sum(logpdf(dph, d) for d in delta)
    return (ll = ll, terms = terms)
end

@model function hagelloch_variant(cells, pairs)
    beta_C1 ~ Exponential(0.2)
    beta_C2 ~ Exponential(0.1)
    beta_H  ~ Exponential(0.1)
    beta_G  ~ Exponential(0.1)
    mean_δ  ~ Uniform(δ_floor_ip + 0.5, 25.0)
    propagate = glct_propagate(beta_C1, beta_C2, beta_H, beta_G)
    logmarg, lps = marginal_r_loglik(cells; threaded = true) do (nb_r, N2)
        cell_ll(propagate, nb_r, N2, mean_δ, pairs[N2]).ll
    end
    Turing.@addlogprob! logmarg
    return lps
end

# ---- check: at the current bounds the variant likelihood differs from the notebook's by a constant
function check_current()
    cells_c, pairs_c = variant_support(:current)
    rng = Xoshiro(11); diffs = Float64[]
    for _ in 1:8
        b = (0.05 + 0.3rand(rng), 0.02 + 0.1rand(rng), 0.01 + 0.05rand(rng), 0.001 + 0.01rand(rng)); md = 6 + 4rand(rng)
        nb, _ = marginal_r_loglik(cells) do (nb_r, N2)
            hagelloch_cell(glct_propagate(b...), nb_r, N2, md, δ_floor_ip, N0_grid, N1_grid, K2, th0, th1, th2, delta).ll
        end
        vv, _ = marginal_r_loglik(cells_c) do (nb_r, N2); cell_ll(glct_propagate(b...), nb_r, N2, md, pairs_c[N2]).ll; end
        push!(diffs, vv - nb)
    end
    @printf("check (current bounds): variant − notebook log-likelihood = %.10f … %.10f (constant = log 100 = %.10f)\n",
            minimum(diffs), maximum(diffs), log(100)); flush(stdout)
    maximum(diffs) - minimum(diffs) < 1e-8 || error("variant likelihood does not reproduce the notebook's")
end

q(v) = (median(v), quantile(v, 0.025), quantile(v, 0.975))
fmt(v; d = 3) = (x = q(v); string(round(x[1], digits = d), " [", round(x[2], digits = d), ", ", round(x[3], digits = d), "]"))

function run_variant(v)
    cells_v, pairs_v = variant_support(v)
    model = hagelloch_variant(cells_v, pairs_v)
    _ = sample(model, sampler, 1; progress = false, chain_type = MCMCChains.Chains)
    t = @elapsed ch = sample(model, sampler, MCMCThreads(), n_samples, n_chains; progress = false,
                             chain_type = MCMCChains.Chains, initial_params = data_inits(n_chains))
    lps = returned(model, ch); rng = Xoshiro(hash(("hag_sens", v)))
    n_iter, n_ch = size(lps); out = Dict(s => Float64[] for s in (:r, :N0, :N1, :N2, :R0, :tau0, :tau1, :tau2))
    for c in 1:n_ch, i in 1:n_iter
        nb_r, N2 = cells_v[sample_index(rng, lps[i, c])]
        β = [ch[s][i, c] for s in (:beta_C1, :beta_C2, :beta_H, :beta_G)]; md = ch[:mean_δ][i, c]
        cl = cell_ll(glct_propagate(β...), nb_r, N2, md, pairs_v[N2])
        N0, N1 = pairs_v[N2][sample_index(rng, cl.terms)]
        nb_p = nb_r / (md - δ_floor_ip + nb_r)
        sim = glct_propagate(β...)(PhaseTypeDistributions.dph_negative_binomial_min_support(nb_r, nb_p, δ_floor_ip), 1.0 / N2)
        push!(out[:r], nb_r); push!(out[:N0], N0); push!(out[:N1], N1); push!(out[:N2], N2)
        push!(out[:R0], compute_R0_mtheta(β..., md, N0, N1, N2))
        push!(out[:tau0], sim.tau0); push!(out[:tau1], sim.tau1); push!(out[:tau2], sim.tau2)
    end
    s = summarystats(ch[[:beta_C1, :beta_C2, :beta_H, :beta_G, :mean_δ]])
    Ntot = out[:N0] .+ out[:N1] .+ out[:N2]
    lines = [
        "== variant: $v  ($(length(cells_v)) (r, N2) cells; $(round(t, digits = 0)) s sampling)",
        "R0          $(fmt(out[:R0]))",
        "beta_C1     $(fmt(vec(ch[:beta_C1]), d = 4))",
        "beta_C2     $(fmt(vec(ch[:beta_C2]), d = 4))",
        "beta_H      $(fmt(vec(ch[:beta_H]), d = 4))",
        "beta_G      $(fmt(vec(ch[:beta_G]), d = 5))",
        "mean IP     $(fmt(vec(ch[:mean_δ]), d = 2)) d",
        "P(r = 1..4) $(round.([mean(out[:r] .== r) for r in 1:nb_r_max], digits = 3))",
        "N0, N1, N2  $(fmt(out[:N0], d = 0)), $(fmt(out[:N1], d = 0)), $(fmt(out[:N2], d = 0))",
        "N0+N1+N2    $(fmt(Ntot, d = 0));  P(sum = $(N_JOINT)) = $(round(mean(Ntot .== N_JOINT), digits = 3))",
        "attack rate (model τ) class 0/1/2: $(fmt(out[:tau0])) / $(fmt(out[:tau1])) / $(fmt(out[:tau2]));  observed K/N: see N",
        "P(beta_C1 > beta_C2) = $(round(mean(vec(ch[:beta_C1]) .> vec(ch[:beta_C2])), digits = 3))",
        "max R̂ $(round(maximum(s[:, :rhat]), digits = 4)), min bulk ESS $(round(minimum(s[:, :ess_bulk]), digits = 0)), divergences $(Int(sum(vec(ch[:numerical_error]))))",
    ]
    open(joinpath(OUTD, "sensitivity_N_bounds.txt"), "a") do io; foreach(l -> println(io, l), lines); println(io); end
    foreach(println, lines); println(); flush(stdout)
    return ch
end

# compute_R0_mtheta, copied from the notebook's R0 cell (that cell also summarises the notebook's chains)
function compute_R0_mtheta(beta_C1, beta_C2, beta_H, beta_G, mean_ip, N0, N1, N2)
    K = mean_ip .* [
        (N0/N0)*beta_G            (N0/N1)*(beta_H + beta_G)   (N0/N2)*(beta_H + beta_G);
        (N1/N0)*beta_G            (N1/N1)*(beta_C1 + beta_G)  (N1/N2)*beta_G;
        (N2/N0)*beta_G            (N2/N1)*beta_G               (N2/N2)*(beta_C2 + beta_G)
    ]
    return maximum(real(eigvals(K)))
end

if IS_MAIN
    println("K = $K0/$K1/$K2; notebook bounds N_max = $N0_max/$N1_max/$N2_max; joint total ≤ $N_JOINT"); flush(stdout)
    check_current()
    for v in Symbol.(filter(!isempty, split(get(ENV, "VARIANTS", "joint,fixed"), ",")))
        run_variant(v)
    end
end
