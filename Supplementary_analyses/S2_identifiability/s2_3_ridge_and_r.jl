# Supplement S2 (Figure S1; P(r) and KL in the text): R0–mean-IP ridge with the discrete Euler–Lotka curve; P(r) against the prior, KL.
# Reads chains written by s2_2_corroborations.jl (TASK=wide, misc). Usage: julia --project=@. S2_identifiability/s2_3_ridge_and_r.jl
# (A) Joint posterior draws of R0 vs mean IP for rungs 1, 2, 6 with the discrete Euler–Lotka curve
#     R0 = 1 / Σ_j g(j) e^{-λ j},  g(j) = P(W > j-1) / E[W]  (generation interval of the DPH map)
# (B) P(r): posterior vs effective prior for rungs 1–6, with KL(posterior || prior)
ENV["GKSwstype"] = "100"
include(joinpath(@__DIR__, "s2_model.jl"))
using Serialization, Plots
gr()
const OUT = open(joinpath(RES, "s2_3_ridge_r.txt"), "w")
say(x) = (println(x); println(OUT, x); flush(OUT))
const FIG = FIGS
ld(tag) = deserialize(joinpath(CHAINS, "$tag.jls"))

# ---------------------------------------------------------------- generation interval & Euler–Lotka
"""Survivor P(W > a), a = 0..A, for NegBin(r, p) total-trials DPH (W ≥ r)."""
surv(r, p, A) = [cdf(Binomial(a, p), r - 1) for a in 0:A]          # P(fewer than r successes in a trials)
function el_sum(r, μ, λ; A = 3000)                                  # Σ_j g(j) e^{-λ j}
    S = surv(r, r / μ, A)
    return sum(S[a+1] * exp(-λ * (a + 1)) for a in 0:A) / μ
end
R0_theory(r, μ, λ) = 1 / el_sum(r, μ, λ)
function growth_rate(r, μ, R0)                                      # solve R0 · Σ g e^{-λj} = 1
    lo, hi = -0.3, 2.0
    for _ in 1:80
        mid = (lo + hi) / 2
        R0 * el_sum(r, μ, mid; A = 600) > 1 ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end
# truth: Erlang-4 DPH from the simulator (generic DPH survivor α'T^a 1)
S_true = let x = copy(dph_nom.α), out = Float64[]
    for _ in 0:600; push!(out, sum(x)); x = dph_nom.S' * x; end; out  # α' T^a 1, advanced as in glct_map (x ← T'x)
end
S_true2 = [sum((dph_nom.α' * dph_nom.S^a)) for a in 0:60]
@assert maximum(abs.(S_true[1:61] .- S_true2)) < 1e-10
λ_true = let lo = -0.3, hi = 2.0
    for _ in 1:80
        mid = (lo + hi) / 2
        β_true * sum(S_true[a+1] * exp(-mid * (a + 1)) for a in 0:600) > 1 ? (lo = mid) : (hi = mid)
    end
    (lo + hi) / 2
end
say("True early growth rate λ = $(round(λ_true, digits=4)) /day (doubling time $(round(log(2)/λ_true, digits=2)) d); truth R0 = $(round(R0_imp, digits=3)), E[W] = $(round(EW, digits=3))")
# consistency: Euler–Lotka with the NegBin(4, 4/E[W]) curve at the truth
say("Euler–Lotka check: R0_theory(r=4, μ=E[W], λ_true) = $(round(R0_theory(4, EW, λ_true), digits=4)) (NegBin DPH with the truth mean; truth DPH is Erlang-from-mean, so close but not identical)")

# ---------------------------------------------------------------- per-rung draws
specs = Dict(1 => (:discrete, :multbin, false), 2 => (:discrete, :pois, false), 3 => (:discrete, :nb, false),
             4 => (:ode, :pois, false), 5 => (:ode, :nb, false), 6 => (:discrete, :multbin, true))
function draws(tag, rung, hi)
    R0_HI[] = hi
    m = ladder(specs[rung]...); ch = ld(tag)
    lps = returned(m, ch)
    return (R0 = vec(ch[:R0]), ip = vec(ch[:mean_ip]), r = vec([sample_index(l) for l in lps]),
            Pr = vec(mean(reduce(hcat, [posterior_weights(l) for l in vec(lps)]), dims = 2)))
end

# ---------------------------------------------------------------- (A) ridge plot
cols = Dict(1 => "#2a78d6", 2 => "#eb6834", 6 => "#1baf7a")
shapes = Dict(1 => :circle, 2 => :utriangle, 6 => :diamond)
labs = Dict(1 => "Infection-only DDSA", 2 => "Map–Poisson", 6 => "Full-data DDSA")   # model names used in the supplement
μgrid = range(1.2, 30, length = 600)
panels = []
Random.seed!(7)
for (hi, tagf) in ((5.0, r -> "trunc_rung$r"), (20.0, r -> "wide_rung$r"))
    p = plot(xlabel = "Mean infectious period (days)", ylabel = "R₀", title = "R₀ prior U(0.5, $(Int(hi)))",
             xlims = (0, hi == 5 ? 16 : 30.5), ylims = (0, hi == 5 ? 5.3 : 20.5), legend = hi == 5 ? :bottomright : false,
             grid = true, gridalpha = 0.15, framestyle = :box, titlefontsize = 10, guidefontsize = 9,
             tickfontsize = 8, legendfontsize = 9, foreground_color_legend = nothing)
    for (rr, ls, lab) in ((1, :dot, "Euler–Lotka, r = 1"), (4, :solid, "Euler–Lotka, r = 4"), (10, :dash, "Euler–Lotka, r = 10"))
        xs = [μ for μ in μgrid if μ > rr]
        plot!(p, xs, [R0_theory(rr, μ, λ_true) for μ in xs], color = "#55534e", lw = 1.5, ls = ls, label = lab)
    end
    for rung in (2, 1, 6)
        d = draws(tagf(rung), rung, hi)
        idx = randperm(length(d.R0))[1:min(2500, length(d.R0))]
        scatter!(p, d.ip[idx], d.R0[idx], ms = 2.2, msw = 0, alpha = 0.35, color = cols[rung],
                 marker = shapes[rung], label = labs[rung])
        λd = [growth_rate(d.r[j], d.ip[j], d.R0[j]) for j in idx[1:800]]
        say("R0≤$(Int(hi)) rung $rung: implied growth rate λ = $(q(λd)) /day; R0 = $(q(d.R0)); mean IP = $(q(d.ip)); " *
            "corr(log R0, log IP) = $(round(cor(log.(d.R0), log.(d.ip)), digits=3))")
    end
    scatter!(p, [EW], [R0_imp], marker = :star5, ms = 8, color = :black, label = "Truth")
    push!(panels, p)
end
pA = plot(panels..., layout = (1, 2), size = (980, 420), left_margin = 4Plots.mm, bottom_margin = 4Plots.mm, dpi = 200)
savefig(pA, joinpath(FIG, "s2_fig_ridge_R0_meanIP.png")); savefig(pA, joinpath(FIG, "s2_fig_ridge_R0_meanIP.pdf"))

# ---------------------------------------------------------------- (B) P(r) vs prior, KL
# Effective prior: r ~ U{1..10}, μ ~ U(1, 30), support r/μ < 1  ⇒  P(r) ∝ P(μ > r) = (30 − r)/29
prior_r = [(30 - r) / 29 for r in 1:r_max]; prior_r ./= sum(prior_r)
KL(p, q) = sum(p[i] > 0 ? p[i] * log(p[i] / q[i]) : 0.0 for i in eachindex(p))
say("\nEffective prior P(r) = $(round.(prior_r, digits=3)); max possible KL (posterior = point mass at r=4) = $(round(-log(prior_r[4]), digits=3)) nats")
rnames = Dict(1 => "1 DDSA Scen. B", 2 => "2 Discrete, Poisson", 3 => "3 Discrete, NegBin",
              4 => "4 ODE, Poisson", 5 => "5 ODE, NegBin", 6 => "6 Full line list")
bars = []
for rung in 1:6
    d = draws("wide_rung$rung", rung, 20.0)
    kl = KL(d.Pr, prior_r)
    ent = -sum(x > 0 ? x * log(x) : 0.0 for x in d.Pr)
    say("R0≤20 rung $(rnames[rung]): P(r) = $(round.(d.Pr, digits=3)) | KL(post‖prior) = $(round(kl, digits=3)) nats | entropy $(round(ent, digits=3)) (prior $(round(-sum(prior_r .* log.(prior_r)), digits=3)))")
    b = bar(1:r_max, d.Pr, color = rung == 6 ? "#1baf7a" : "#2a78d6", lc = :white, lw = 1, bar_width = 0.75,
            label = "Posterior", ylims = (0, rung == 6 ? 1.05 : 0.32), xticks = 1:r_max,
            title = "$(rnames[rung])  (KL = $(round(kl, digits=2)) nats)", titlefontsize = 9, tickfontsize = 7,
            guidefontsize = 8, legendfontsize = 7, framestyle = :box, grid = :y, gridalpha = 0.15,
            xlabel = rung >= 4 ? "Phase count r" : "", ylabel = rung in (1, 4) ? "Probability" : "",
            legend = rung == 1 ? :topright : false, foreground_color_legend = nothing)
    scatter!(b, 1:r_max, prior_r, marker = :hline, ms = 9, msw = 2, color = "#55534e", label = "Effective prior")
    push!(bars, b)
end
# R0 ≤ 5 fits saved for rungs 1, 2, 6 only
for (rung, tag) in ((1, "trunc_rung1"), (2, "trunc_rung2"), (6, "trunc_rung6"))
    d = draws(tag, rung, 5.0)
    say("R0≤5  rung $(rnames[rung]): P(r) = $(round.(d.Pr, digits=3)) | KL(post‖prior) = $(round(KL(d.Pr, prior_r), digits=3)) nats")
end
pB = plot(bars..., layout = (2, 3), size = (980, 520), left_margin = 3Plots.mm, bottom_margin = 3Plots.mm, dpi = 200)
savefig(pB, joinpath(FIG, "s2_fig_Pr_vs_prior.png")); savefig(pB, joinpath(FIG, "s2_fig_Pr_vs_prior.pdf"))
close(OUT)
