# Supplement S2: numerical grid posteriors of the three models without recovery intervals on the datasets of
# s2_traj_data.jl: N = 99 (ladder data) and N = 1,000 and 10,000 (the attack-rate sweep generator at R0 = 3.16). The draws
# are read by s2_9_trajectories.jl, which fits full-data DDSA and draws the trajectory figures.
#   map–Poisson          discrete map with a NegBin DPH infectious period, Poisson daily counts      (ladder model 2)
#   ODE–Poisson          continuous Erlang-r ODE (the NegBin DPH's continuous analogue), Poisson daily counts (ladder
#                        model 4, with r = 1..R_ODE and no constraint μ > r; see Models)
#   infection-only DDSA  discrete map, Multinomial infection days + Binomial(N, τ) final size     (ladder model 1)
# Infection-only DDSA and map–Poisson differ only in how the number of cases is scored (Binomial(N, τ) against
# Poisson(Nτ)); comparing them with full-data DDSA separates the contribution of the final-size term from that of the
# recovery intervals.
# With counts only, the NUTS fits we ran did not converge at N ≥ 1,000 (map–Poisson in s2_4; NegBin count models in an
# earlier version of s2_9_trajectories.jl): the counts identify the early growth rate and the timing but not R0 and the mean
# IP separately, so the posterior is a long, thin ridge whose width shrinks with N. After summing out r each model has three
# continuous parameters (R0, μ, ρ), so the posterior is computed on a grid instead.
# Models. Priors as the ladder with R0 ~ U(0.5, 20): μ ~ U(1, 30), ρ ~ Beta(1, N − 1). Map models: r uniform on 1..10 with
# r < μ (p = r/μ < 1), as `ladder(:discrete, :pois | :multbin, false)`. ODE: r uniform on 1..R_ODE = 40 with no constraint
# on μ (see s2_traj_data.jl). For the ODE, the mean interval that daily recording would give, E[max(1, ⌊t+W⌋ − ⌊t⌋)] =
# μ + E[(1 − W)⁺] for an infection time t uniform within its day, is also reported, to compare with the data's mean
# recorded interval (6.33 d).
# N is the whole population, including the initial infectives.
# Grid. μ: 0.025 d steps below 15 d, 0.25 d above (as s2_5). For each (μ, r), the conditional posterior of (R0, log ρ) is a
# narrow ellipse (at N = 10,000 its thinnest direction has sd ≈ 0.004), so it is integrated on its own window: a coarse pass
# over the whole prior range (R0 step 0.05, log ρ step 0.2) finds it, Newton steps find the conditional mode and its
# curvature, and a 49 × 49 grid spans ±7 conditional sd in R0 and log ρ, clipped to the prior. If the Newton steps fail
# (mode at a prior bound), the window is the box of coarse cells within 25 log units of the coarse maximum. Reported
# diagnostics: the largest share of a window's mass in its outer cells, and coarse cells outside the window within 10 log
# units of the coarse maximum (a second local mode). Quadrature: midpoint rule, each (μ, r) weighted by its μ step and each
# window cell by its area. REFINE=2 halves the μ steps and doubles the points per window axis (WIN_REFINE=1 keeps the
# windows); MU_RANGE=lo,hi restricts μ to [lo, hi] (refinement checks over the region holding the posterior mass).
# Checks. Grid log density = Turing log-joint at random points (as s2_7): `ladder(:discrete, :pois, false)` for the map,
# `ode_poisson()` below for the ODE, `ladder(:discrete, :multbin, false)` for infection-only DDSA. At N = 1,000 the counts
# are those of s2_5's R0 = 3.16 cell (grid posteriors: map–Poisson R0 4.61 [2.72, 12.51], mean IP 10.71 [4.89, 29.0];
# infection-only DDSA R0 3.17 [2.93, 3.4], mean IP 6.44 [4.91, 7.74]).
# Usage (from the supplementary folder): julia --project=@. -t 10 S2_identifiability/s2_8_grid_poisson.jl
#   GRID_MODEL=map,ode,inf (default) and GRID_N=99,1000,10000 (default) select the fits; REFINE=2 runs the accuracy check.
#   Run times on 10 threads: map and inf 3–4 min per N; ODE about 1.5–2 h per N (24,800 (μ, r) cells, one ODE solve per point).
# Outputs: out/s2_8_grid_{map,ode}_poisson_N{N}[_refine2].txt, out/s2_8_grid_infonly_N{N}[…].txt; chains/grid_… .jls with
# the same stems (4,000 posterior draws of (R0, mean_ip, ρ, r), with R0 and ρ jittered within their window cell and mean_ip
# at its grid point, plus the data K and th; read by s2_9_trajectories.jl for the trajectories of these models).
include(joinpath(@__DIR__, "s2_model.jl"))
include(joinpath(@__DIR__, "s2_traj_data.jl"))   # DATA99, set_data!, R_ODE, rsupport
using Serialization, Printf, ForwardDiff, Dates
R0_HI[] = 20.0
const GMODELS = Symbol.(split(get(ENV, "GRID_MODEL", "map,ode"), ","))
const NS   = parse.(Int, split(get(ENV, "GRID_N", "99,1000,10000"), ","))
const REF  = parse(Int, get(ENV, "REFINE", "1"))
const WREF = parse(Int, get(ENV, "WIN_REFINE", string(REF)))
const MUR  = isempty(get(ENV, "MU_RANGE", "")) ? (1.0, 30.0) : Tuple(parse.(Float64, split(ENV["MU_RANGE"], ",")))
const SUF  = (REF > 1 ? "_refine$(REF)" : "") * (WREF != REF ? "_win$(WREF)" : "") *
             (MUR == (1.0, 30.0) ? "" : "_mu$(MUR[1])-$(MUR[2])")
const NW   = 49WREF                      # points per window axis
const WSD  = 7.0                         # window half-width in conditional sd
const NDRAW = 4000
lse(v) = (m = maximum(v); isfinite(m) ? m + log(sum(exp.(v .- m))) : m)

# ---------------------------------------------------------------- log density
"""Log-likelihood of the infection data for model m and stage count r: Poisson daily counts (:map, :ode) or the
Multinomial + Binomial terms of infection-only DDSA (:inf)."""
function ll_pois(m, r, R0, μ, ρ)
    m === :map && return ll_rung(r, R0, μ, ρ, 0.0; map = :discrete, obs = :pois, recov = false)
    m === :inf && return ll_rung(r, R0, μ, ρ, 0.0; map = :discrete, obs = :multbin, recov = false)
    μt = mu_ode(r, R0, μ, ρ)
    return μt === nothing ? -Inf : ll_counts(μt, :pois, 0.0)
end
"""Log density of (R0, log ρ) for fixed (μ, r), with the priors of R0, μ, ρ and r and the Jacobian ρ of the log ρ scale."""
function lpost(m, r, R0, μ, lρ)
    (0.5 < R0 < R0_HI[] && lρ < 0) || return -Inf
    ρ = exp(lρ)
    ll = ll_pois(m, r, R0, μ, ρ)
    return ll - log(length(rsupport(m))) + logpdf(Uniform(0.5, R0_HI[]), R0) + logpdf(Uniform(1.0, 30.0), μ) + logpdf(Beta(1.0, N - 1), ρ) + lρ
end
"""Grid-form log density at (R0, μ, ρ) on the ρ scale (as s2_7), for the check against Turing."""
grid_logdensity(m, R0, μ, ρ) = lse([lpost(m, r, R0, μ, log(ρ)) - log(ρ) for r in rsupport(m)])
"""ODE–Poisson as a Turing model (r summed out), for the check of the grid log density."""
@model function ode_poisson()
    R0 ~ Uniform(0.5, R0_HI[])
    mean_ip ~ Uniform(1.0, 30.0)
    ρ ~ Beta(1.0, N - 1)
    logmarg, _ = marginal_r_loglik(r -> ll_pois(:ode, r, R0, mean_ip, ρ), rsupport(:ode))
    Turing.@addlogprob! logmarg
end

# ---------------------------------------------------------------- per-(μ, r) window
"""Integrate exp(lpost) over (R0, log ρ) for fixed (μ, r). Returns (logZ, window, edge share, fallback used, second-mode cells)."""
function cond_window(m, r, μ, R0c, lρc)
    C = [lpost(m, r, R0, μ, lρ) for R0 in R0c, lρ in lρc]
    cmax = maximum(C)
    isfinite(cmax) || return (logZ = -Inf, win = nothing, edge = 0.0, fallback = false, second = 0)
    ib = argmax(C); x = [R0c[ib[1]], lρc[ib[2]]]
    f(y) = lpost(m, r, y[1], μ, y[2])
    ok = false
    for _ in 1:40
        H = ForwardDiff.hessian(f, x)
        (all(isfinite, H) && isposdef(-Symmetric(H))) || break
        s = -(Symmetric(H) \ ForwardDiff.gradient(f, x)); fx = f(x); a = 1.0
        while a > 1e-4 && !(f(x .+ a .* s) >= fx); a /= 2; end
        a > 1e-4 || break
        x = x .+ a .* s
        if maximum(abs.(a .* s)) < 1e-7
            ok = true; break
        end
    end
    H = ok ? ForwardDiff.hessian(f, x) : nothing
    ok = ok && all(isfinite, H) && isposdef(-Symmetric(H))
    if ok
        Σ = inv(-Symmetric(H)); sR, sρ = sqrt(Σ[1, 1]), sqrt(Σ[2, 2])
        Rlo, Rhi = max(0.5, x[1] - WSD * sR), min(R0_HI[], x[1] + WSD * sR)
        ρlo, ρhi = x[2] - WSD * sρ, min(0.0, x[2] + WSD * sρ)
    else                                  # box of coarse cells within 25 log units of the coarse maximum, padded by a cell
        keep = findall(C .> cmax - 25)
        dRc, dρc = R0c[2] - R0c[1], lρc[2] - lρc[1]
        Rlo = max(0.5, minimum(R0c[i[1]] for i in keep) - dRc); Rhi = min(R0_HI[], maximum(R0c[i[1]] for i in keep) + dRc)
        ρlo = minimum(lρc[i[2]] for i in keep) - dρc; ρhi = min(0.0, maximum(lρc[i[2]] for i in keep) + dρc)
    end
    dR, dρ = (Rhi - Rlo) / NW, (ρhi - ρlo) / NW
    Rs = Rlo .+ dR .* ((1:NW) .- 0.5); ρs = ρlo .+ dρ .* ((1:NW) .- 0.5)
    L = [lpost(m, r, R0, μ, lρ) for R0 in Rs, lρ in ρs]
    lz = lse(vec(L))
    isfinite(lz) || return (logZ = -Inf, win = nothing, edge = 0.0, fallback = !ok, second = 0)
    w = exp.(L .- lz)
    edge = sum(w[1, :]) + sum(w[end, :]) + sum(w[2:end-1, 1]) + sum(w[2:end-1, end])
    # coarse cells outside the window that are within 10 log units of the coarse maximum: a second local mode
    second = count(i -> C[i] > cmax - 10 && !(Rlo <= R0c[i[1]] <= Rhi && ρlo <= lρc[i[2]] <= ρhi), CartesianIndices(C))
    return (logZ = lz + log(dR) + log(dρ), win = (Rlo = Rlo, dR = dR, ρlo = ρlo, dρ = dρ, L = L), edge = edge, fallback = !ok, second = second)
end

# ---------------------------------------------------------------- grid posterior
wq(x, w, p) = (o = sortperm(x); c = cumsum(w[o]); x[o][min(searchsortedfirst(c, p), length(x))])
qw(x, w) = "$(round(wq(x, w, 0.5), digits = 2)) [$(round(wq(x, w, 0.025), digits = 2)), $(round(wq(x, w, 0.975), digits = 2))]"
μs = vcat(collect(range(1.0 + 0.0125 / REF, 15.0 - 0.0125 / REF, length = 560REF)), collect(range(15.0 + 0.125 / REF, 30.0 - 0.125 / REF, length = 60REF)))
dμ = vcat(fill(0.025 / REF, 560REF), fill(0.25 / REF, 60REF))
keepμ = MUR[1] .<= μs .<= MUR[2]; μs = μs[keepμ]; dμ = dμ[keepμ]
"""Mean recorded interval E[max(1, ⌊t+W⌋ − ⌊t⌋)] = μ + E[(1 − W)⁺] for W ~ Erlang(r, mean μ) and t ~ U(0, 1)."""
rec_mean(r, μ) = μ + sum(cdf(Gamma(r, μ / r), u) for u in ((1:2000) .- 0.5) ./ 2000) / 2000

for m in GMODELS, N_s in NS
    name = m === :map ? "map–Poisson" : m === :ode ? "ODE–Poisson" : "infection-only DDSA"
    stem = m === :inf ? "grid_infonly_N$(N_s)$(SUF)" : "grid_$(m)_poisson_N$(N_s)$(SUF)"
    RS = rsupport(m)
    out = open(joinpath(RES, "s2_8_$(stem).txt"), "w")
    say(x) = (println(x); println(out, x); flush(out))
    β_s, ρ_s = set_data!(N_s)
    say("== $name, numerical grid posterior, N=$(N_s) (whole population), K=$K, T=$nsteps d ($(Dates.now()))")
    say("   truth: R0=$(round(β_s * EW, digits = 3)), mean IP E[W]=$(round(EW, digits = 3)) d, ρN=$(round(ρ_s * N_s, digits = 2)); " *
        "priors R0 ~ U(0.5, $(R0_HI[])), μ ~ U(1, 30), ρ ~ Beta(1, N − 1), " *
        (m === :ode ? "r uniform on 1..$R_ODE (Erlang stages, no constraint on μ)" : "r uniform on 1..$r_max with r < μ") *
        (MUR == (1.0, 30.0) ? "" : "; μ restricted to [$(MUR[1]), $(MUR[2])]"))
    # check: grid log density = Turing log-joint (as s2_7)
    tmodel = m === :map ? ladder(:discrete, :pois, false) : m === :inf ? ladder(:discrete, :multbin, false) : ode_poisson()
    rng = Xoshiro(2026); maxd = 0.0; npts = 0
    for _ in 1:25
        R0 = 0.6 + 19rand(rng); μ = 1.5 + 28rand(rng); ρ = 10^(log10(0.1 / N_s) + 2rand(rng))
        g = grid_logdensity(m, R0, μ, ρ); isfinite(g) || continue
        lj = Turing.logjoint(tmodel, (R0 = R0, mean_ip = μ, ρ = ρ))
        maxd = max(maxd, abs(lj - g)); npts += 1
    end
    say(@sprintf("   check: %d random points, max |Turing log-joint − grid log density| = %.2e", npts, maxd))

    R0c = collect(0.525:0.05:19.975); lρc = collect(range(log(0.03 / N_s), log(30 / N_s), length = 36))
    res = Matrix{Any}(undef, length(μs), length(RS))
    t = @elapsed Threads.@threads :greedy for iμ in eachindex(μs)   # one μ at a time per free thread (cost varies with μ)
        for r in RS
            res[iμ, r] = (m === :ode || r < μs[iμ]) ? cond_window(m, r, μs[iμ], R0c, lρc) :
                         (logZ = -Inf, win = nothing, edge = 0.0, fallback = false, second = 0)
        end
    end
    lw = [res[i, r].logZ + log(dμ[i]) for i in eachindex(μs), r in RS]
    W = exp.(lw .- lse(vec(lw))); W ./= sum(W)
    wμ = vec(sum(W, dims = 2)); wr = vec(sum(W, dims = 1))
    fin = [(i, r) for i in eachindex(μs), r in RS if isfinite(lw[i, r])]
    mass(f) = sum(W[i, r] for (i, r) in fin if f(res[i, r]); init = 0.0)
    say(@sprintf("   grid: %d μ × %d r, windows %d × %d (±%.0f sd), %d finite (μ, r) cells, %.0f s on %d threads",
                 length(μs), length(RS), NW, NW, WSD, length(fin), t, Threads.nthreads()))
    say(@sprintf("   diagnostics: max window edge share %.1e (mass-weighted mean %.1e) | fallback windows %d (posterior mass %.1e) | windows with second-mode coarse cells %d (posterior mass %.1e)",
                 maximum(res[i, r].edge for (i, r) in fin), sum(W[i, r] * res[i, r].edge for (i, r) in fin),
                 count(((i, r),) -> res[i, r].fallback, fin), mass(x -> x.fallback),
                 count(((i, r),) -> res[i, r].second > 0, fin), mass(x -> x.second > 0)))
    fb = sort(filter(((i, r),) -> res[i, r].fallback, fin), by = ((i, r),) -> -W[i, r])
    for (i, r) in fb[1:min(3, end)]
        v = res[i, r].win
        say(@sprintf("      fallback window μ=%.4f r=%d: mass %.1e, R0 [%.2f, %.2f], ρN [%.2g, %.2g], edge share %.1e", μs[i], r, W[i, r],
                     v.Rlo, v.Rlo + NW * v.dR, exp(v.ρlo) * N, exp(v.ρlo + NW * v.dρ) * N, res[i, r].edge))
    end
    # marginal quantiles of R0 and ρN over all window cells (cell centres, weight = (μ, r) weight × cell weight)
    xs = Float64[]; ys = Float64[]; ws = Float64[]
    for (i, r) in fin
        v = res[i, r].win; p = exp.(v.L .- lse(vec(v.L))) .* W[i, r]
        for k in CartesianIndices(v.L)
            p[k] > 1e-14 || continue
            push!(xs, v.Rlo + v.dR * (k[1] - 0.5)); push!(ys, exp(v.ρlo + v.dρ * (k[2] - 0.5)) * N); push!(ws, p[k])
        end
    end
    ws ./= sum(ws)

    # draws: (μ, r) by weight, then a window cell by weight, jittered uniformly within the cell
    Random.seed!(7); cw = cumsum(vec(W)); cart = CartesianIndices(W)
    R0d = zeros(NDRAW); μd = zeros(NDRAW); ρd = zeros(NDRAW); rdr = zeros(Int, NDRAW)
    for j in 1:NDRAW
        c = cart[min(searchsortedfirst(cw, rand()), length(cw))]; i, r = c[1], c[2]; v = res[i, r].win
        p = exp.(vec(v.L) .- lse(vec(v.L))); k = CartesianIndices(v.L)[min(searchsortedfirst(cumsum(p), rand()), length(p))]
        R0d[j] = v.Rlo + v.dR * (k[1] - rand()); ρd[j] = exp(v.ρlo + v.dρ * (k[2] - rand())); μd[j] = μs[i]; rdr[j] = r
    end
    recd = m === :ode ? [rec_mean(rdr[j], μd[j]) for j in 1:NDRAW] : copy(μd)     # the map's mean IP is already a daily one
    serialize(joinpath(CHAINS, "$(stem).jls"), (R0 = R0d, mean_ip = μd, ρ = ρd, r = rdr, rec_ip = recd, K = K, th = copy(th)))
    cases = [sum(m === :ode ? mu_ode(rdr[j], R0d[j], μd[j], ρd[j]) : mu_discrete(rdr[j], R0d[j], μd[j], ρd[j])[1]) for j in 1:NDRAW]
    say("   posterior: R0 = $(qw(xs, ws)) | meanIP = $(qw(μs, wμ)) d | ρN = $(qw(ys, ws)) | P(meanIP < 5) = $(round(sum(wμ[μs .< 5]), digits = 3)) | " *
        "P(meanIP > 20) = $(round(sum(wμ[μs .> 20]), digits = 3)) | P(R0 > 10) = $(round(sum(ws[xs .> 10]), digits = 3))")
    say("          P(r) = $(round.(wr, digits = 3))")
    say("   draws ($NDRAW): R0 = $(q(R0d)) | meanIP = $(q(μd)) d | ρN = $(q(ρd .* N)) | expected cases Στ·N = $(q(cases)) (observed K = $K)")
    m === :ode && say("   recorded mean interval under daily recording, μ + E[(1 − W)⁺]: $(q(recd)) d (data's mean recorded interval $(round(mean(delta), digits = 2)) d; DPH mean $(round(EW, digits = 3)) d)")
    (m === :map && N_s == 1000) && say("   s2_5 (same counts, R0 = 3.16 cell, global grid): R0 = 4.61 [2.72, 12.51] | meanIP = 10.71 [4.89, 29.0] | " *
                                       "P(meanIP < 5) = 0.03 | P(r) = [0.211, 0.227, 0.202, 0.133, 0.082, 0.052, 0.035, 0.025, 0.019, 0.015]")
    (m === :inf && N_s == 1000) && say("   s2_5 (same counts, R0 = 3.16 cell, global grid): R0 = 3.17 [2.93, 3.4] | meanIP = 6.44 [4.91, 7.74]")
    (m === :inf && N_s == 99) && say("   s2_5 (N = 99, R0 = 3.16 cell, global grid): R0 = 3.21 [2.48, 10.26] | meanIP = 6.72 [3.88, 27.62]")
    close(out)
end
