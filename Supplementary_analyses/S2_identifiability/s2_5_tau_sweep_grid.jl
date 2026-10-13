# Supplement S2 (Table S8) and S8 (Figure S2): exact grid posteriors for the attack-rate sweep, compared with MCMC.
# GRID_N=99 or 1000 (default 1000). CELLS=1.5,2.0,3.16 selects true R0 values (default all; the N = 1000 rung-2 cells
# take 5–8 min each, so run them one at a time under a 10-minute cap). Reads chains from s2_4_tau_sweep_mcmc.jl.
# Exact (grid) posterior for the τ-sweep cells. After summing out r, the model has three
# continuous parameters (R0, μ, ρ), so the posterior can be computed on a grid. This resolves the
# mean-IP multimodality that the NUTS chains did not cross in these fits, and gives the weight of each mode.
ENV["GKSwstype"] = "100"
include(joinpath(@__DIR__, "s2_model.jl"))
const GN = parse(Int, get(ENV, "GRID_N", "1000"))
const CELLSEL = haskey(ENV, "CELLS") ? parse.(Float64, split(ENV["CELLS"], ",")) : nothing
const REF = parse(Int, get(ENV, "REFINE", "1"))   # grid-refinement check: steps divided by REF (N = 99 grids only)
const SUF = (CELLSEL === nothing ? "" : "_R0" * join(ENV["CELLS"] |> x -> split(x, ","), "_")) * (REF > 1 ? "_refine$(REF)" : "")
using Serialization, Plots
gr()
const OUT = open(joinpath(RES, "s2_5_tausweep_grid_N$(GN)$(SUF).txt"), "w")
say(x) = (println(x); println(OUT, x); flush(OUT))
const FIG = FIGS
R0_HI[] = 20.0

function truth_SI(β, ρ, T)
    S = 1 - ρ; x = dph_nom.α .* ρ
    for _ in 1:T
        ni = (1 - exp(-β * sum(x))) * S
        x = dph_nom.S' * x + dph_nom.α .* ni; S -= ni
    end
    return S, sum(x)
end
"""Regenerate the τ-sweep dataset exactly as s2_4_tau_sweep_mcmc.jl does and set the globals."""
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

# μ grid: fine where the modes are (boundary modes sit just above integers r, where p = r/μ → 1)
μs, dμ = GN == 1000 ?
    (vcat(collect(1.0125:0.025:15.0), collect(15.25:0.25:30.0)), vcat(fill(0.025, length(1.0125:0.025:15.0)), fill(0.25, length(15.25:0.25:30.0)))) :
    (REF == 1 ? (vcat(collect(1.025:0.05:15.0), collect(15.125:0.25:29.875)), vcat(fill(0.05, length(1.025:0.05:15.0)), fill(0.25, length(15.125:0.25:29.875)))) :
     (vcat(collect(range(1.0 + 0.025 / REF, 15.0 - 0.025 / REF, length = 280REF)), collect(range(15.0 + 0.125 / REF, 30.0 - 0.125 / REF, length = 60REF))),
      vcat(fill(0.05 / REF, 280REF), fill(0.25 / REF, 60REF))))

function grid_posterior(obs, R0s, dRs, ρs)
    nμ, nR, nρ = length(μs), length(R0s), length(ρs)
    lp = fill(-Inf, r_max, nμ, nR, nρ)
    lρ = log.(ρs); dlρ = lρ[2] - lρ[1]
    Threads.@threads for iμ in 1:nμ
        for iR in 1:nR, iρ in 1:nρ
            ρ = ρs[iρ]
            # prior: R0, μ uniform (cell widths dR, dμ); ρ ~ Beta(1, N−1) on a log grid (Jacobian ρ); r uniform
            lpr = log(dRs[iR]) + log(dμ[iμ]) + logpdf(Beta(1.0, N - 1), ρ) + log(ρ * dlρ)
            for r in 1:r_max
                ll = ll_rung(r, R0s[iR], μs[iμ], ρ, 0.0; map = :discrete, obs = obs, recov = false)
                isfinite(ll) && (lp[r, iμ, iR, iρ] = ll + lpr)
            end
        end
    end
    m = maximum(lp); w = exp.(lp .- m); w ./= sum(w)
    return w
end

wq(x, w, p) = (o = sortperm(x); c = cumsum(w[o]); x[o][searchsortedfirst(c, p)])
qw(x, w) = "$(round(wq(x,w,0.5),digits=2)) [$(round(wq(x,w,0.025),digits=2)), $(round(wq(x,w,0.975),digits=2))]"

panels = []
sdτ = Dict()
for R0_t in (1.5, 2.0, R0_imp), (rung, obs) in ((1, :multbin), (2, :pois))
    CELLSEL === nothing || any(abs.(CELLSEL .- round(R0_t, digits=2)) .< 0.005) || continue
    τ_t = set_data!(GN, R0_t)
    ch = deserialize(joinpath(CHAINS, GN == 1000 ? "tau_N1000_R0$(round(R0_t, digits=2))_rung$(rung)_N1000_12ch.jls" : "tau_N99_R0$(round(R0_t, digits=2))_rung$(rung).jls"))
    R0c = vec(ch[:R0]); ipc = vec(ch[:mean_ip]); ρc = vec(ch[:ρ])
    # R0 grid. N = 1000, rung 1 (well-mixed R0): 50 points over the padded MCMC range. Rung 2 (high-R0 modes seen with
    # 12 chains): fine segment around the truth + coarse segments covering the whole prior U(0.5, 20).
    if GN == 99
        # both rungs: whole prior range U(0.5, 20) (N = 99 posteriors are wide; rung 2 has high-R0 mass)
        a, b = max(0.5, R0_t - 0.4), R0_t + 0.6
        seg1 = collect(0.525:0.05:a); fine = collect(a + 0.01:0.02:b); seg3 = collect(b + 0.1:0.2:19.9)
        R0s = vcat(seg1, fine, seg3); dRs = vcat(fill(0.05, length(seg1)), fill(0.02, length(fine)), fill(0.2, length(seg3)))
        if REF > 1      # same segments, each cell split into REF cells
            R0s = vcat([c .+ (((1:REF) .- (REF + 1) / 2) .* (d / REF)) for (c, d) in zip(R0s, dRs)]...); dRs = vcat([fill(d / REF, REF) for d in dRs]...)
        end
    elseif rung == 1
        lo, hi = quantile(R0c, [0.0005, 0.9995]); pad = 0.25 * (hi - lo)
        R0s = collect(range(max(0.5, lo - pad), min(20.0, hi + pad), length = 50)); dRs = fill(R0s[2] - R0s[1], 50)
    else
        a, b = max(0.5, R0_t - 0.4), R0_t + 0.6
        seg1 = collect(0.525:0.05:a); fine = collect(a + 0.005:0.01:b); seg3 = collect(b + 0.05:0.1:19.95)
        R0s = vcat(seg1, fine, seg3); dRs = vcat(fill(0.05, length(seg1)), fill(0.01, length(fine)), fill(0.1, length(seg3)))
    end
    ρs  = exp.(range(log(0.03 / N), log(30 / N), length = GN == 99 ? 35REF + 1 : 36))
    t = @elapsed w = grid_posterior(obs, R0s, dRs, ρs)
    # marginals
    wμ = vec(sum(w, dims = (1, 3, 4))); wR = vec(sum(w, dims = (1, 2, 4))); wr = vec(sum(w, dims = (2, 3, 4)))
    wρ = vec(sum(w, dims = (1, 2, 3)))
    edge = (wR[1] + wR[end], wρ[1] + wρ[end])
    # τ: sample grid cells by weight, compute τ at each
    Random.seed!(2); cells = CartesianIndices(w); cw = cumsum(vec(w))
    idx = [cells[min(searchsortedfirst(cw, rand()), length(cw))] for _ in 1:4000]
    τs = [sum(mu_discrete(i[1], R0s[i[3]], μs[i[2]], ρs[i[4]])[1]) / N for i in idx]
    pA = sum(wμ[μs .< 5.0]); sdτ[(R0_t, rung)] = std(τs)
    GN == 99 && rung == 2 && haskey(sdτ, (R0_t, 1)) && say("   SD(τ) ratio rung 2 / rung 1 = $(round(std(τs) / sdτ[(R0_t, 1)], digits=2)) vs sqrt(1/(1−τ)) = $(round(sqrt(1/(1-τ_t)), digits=2))")
    say("N=$(GN) R0_true=$(round(R0_t,digits=2)) rung $rung (grid $(length(μs))×$(length(R0s))×$(length(ρs))×$r_max, $(round(Int,t)) s):")
    say("   exact: R0 = $(qw(R0s, wR))  P(R0 > $(round(R0_t+0.6,digits=2))) = $(round(sum(wR[R0s .> R0_t + 0.6]), digits=3))  | meanIP = $(qw(μs, wμ))  | τ = $(round(median(τs),digits=3)) [$(round(quantile(τs,0.025),digits=3)), $(round(quantile(τs,0.975),digits=3))] SD(τ) = $(round(std(τs), digits=4))")
    say("          P(meanIP < 5) = $(round(pA, digits=3))   P(r) = $(round.(wr, digits=3))   grid-edge mass: R0 $(round(edge[1], sigdigits=2)), ρ $(round(edge[2], sigdigits=2))")
    say("   MCMC ($(GN == 1000 ? 12 : 4) ch): R0 = $(q(R0c))  | meanIP = $(q(ipc))  | share of draws with meanIP < 5 = $(round(mean(ipc .< 5), digits=3))")
    # figure panel: exact marginal density of mean IP vs pooled MCMC histogram. The bins cover the whole prior range
    # (1–30 d) so that the histogram and the grid density are normalised over the same range; at N = 1000 the axis
    # shows 2–12 d, and the share of each posterior beyond 12 d is printed for the caption.
    GN == 1000 && println("   figure: share of mean IP > 12 d: MCMC $(round(mean(ipc .> 12), digits=3)), grid $(round(sum(wμ[μs .> 12]), digits=3))")
    p = histogram(ipc, bins = GN == 1000 ? range(1, 30, length = 291) : range(1, 30, length = 117), normalize = :pdf, color = "#2a78d6", alpha = 0.45, lw = 0,
                  label = "MCMC, $(GN == 1000 ? 12 : 4) chains", xlims = GN == 1000 ? (2, 12) : (1, 30), title = "R₀ = $(round(R0_t, digits=2)), $(rung == 1 ? "infection-only DDSA" : "map–Poisson")",
                  titlefontsize = 9, tickfontsize = 7, guidefontsize = 8, legendfontsize = 7, framestyle = :box,
                  grid = :y, gridalpha = 0.15, legend = (R0_t == 1.5 && rung == 1) ? :topright : false,
                  xlabel = R0_t > 3 ? "Mean infectious period (days)" : "", ylabel = rung == 1 ? "Density" : "",
                  foreground_color_legend = nothing)
    plot!(p, μs, wμ ./ dμ, color = "#1f1e1c", lw = 1.5, label = "Exact (grid)")
    vline!(p, [EW], color = "#55534e", ls = :dash, lw = 1, label = "Truth E[W]")
    push!(panels, p)
end
pf = plot(panels..., layout = (3, 2), size = (900, 760), left_margin = 3Plots.mm, bottom_margin = 3Plots.mm, dpi = 200)
CELLSEL === nothing && (savefig(pf, joinpath(FIG, "s5_fig_meanIP_exact_vs_mcmc_N$(GN).png")); savefig(pf, joinpath(FIG, "s5_fig_meanIP_exact_vs_mcmc_N$(GN).pdf")))
close(OUT)
