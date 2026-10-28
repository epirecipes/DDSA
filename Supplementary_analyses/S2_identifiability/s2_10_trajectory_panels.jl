# Supplement S2: the fitted trajectories of the four models of s2_9_trajectories.jl at the three population sizes in one
# figure, as the proportion infectious I(t)/N on a common vertical scale (rows: models; columns: N = 99, 1,000, 10,000).
# Each panel shows the posterior median and 95% CrI of the expected proportion infectious (parameter uncertainty only), the
# true mean-field trajectory and the proportion infectious counted from the line list. The draws are those of
# s2_9_trajectories.jl: map–Poisson, ODE–Poisson and infection-only DDSA from the numerical grid posteriors of
# s2_8_grid_poisson.jl, full-data DDSA from the saved NUTS chains of s2_9_trajectories.jl. Nothing is refitted.
# Usage (from the supplementary folder, after s2_8_grid_poisson.jl and s2_9_trajectories.jl):
#   julia --project=@. S2_identifiability/s2_10_trajectory_panels.jl
# Outputs: figs/s2_fig_trajectory_panels.pdf/.png; out/s2_10_prevalence_bands.csv (the plotted values: per N, model and day,
# the 2.5%, 50% and 97.5% posterior quantiles of the expected proportion infectious, the true mean-field value and the
# observed proportion).
ENV["GKSwstype"] = "100"
include(joinpath(@__DIR__, "s2_model.jl"))
include(joinpath(@__DIR__, "s2_traj_data.jl"))   # DATA99, set_data!, truth_SI, rsupport, glct_traj, map_traj, ode_traj, linelist_prev
using Serialization, Plots, Printf
gr()
R0_HI[] = 20.0                                   # the priors of s2_9_trajectories.jl, R0 ~ U(0.5, 20)
const NS    = (99, 1000, 10000)
const NDRAW = 1000                               # posterior draws per band, as in s2_9_trajectories.jl

# rows of the figure; colours as Figure S1 (infection-only DDSA, map–Poisson, full-data DDSA), ODE–Poisson the next slot
const MODELS = ((2, "Map–Poisson",         :discrete, "#eb6834"),
                (4, "ODE–Poisson",         :ode,      "#eda100"),
                (1, "Infection-only DDSA", :discrete, "#2a78d6"),
                (6, "Full-data DDSA",      :discrete, "#1baf7a"))
const GRID = Dict(2 => "grid_map_poisson", 4 => "grid_ode_poisson", 1 => "grid_infonly")

"Parameter draws (r, R0, mean IP, ρ) for one model at the current data, thinned to about NDRAW."
function draws(rung)
    if haskey(GRID, rung)
        f = joinpath(CHAINS, "$(GRID[rung])_N$(N).jls")
        isfile(f) || error("$f not found: run s2_8_grid_poisson.jl with GRID_N=$(N) first")
        g = deserialize(f)
        (haskey(g, :th) && g.K == K && g.th == th) || error("$f was not computed on these data")
        thin = max(1, div(length(g.R0), NDRAW))
        return g.r[1:thin:end], g.R0[1:thin:end], g.mean_ip[1:thin:end], g.ρ[1:thin:end]
    end
    f = joinpath(CHAINS, "traj_N$(N)_rung$rung.jls")
    isfile(f) || error("$f not found: run s2_9_trajectories.jl first")
    ch = deserialize(f)
    rhat = maximum(summarystats(ch[[:R0, :mean_ip, :ρ]])[:, :rhat])
    rhat <= 1.01 || error("full-data DDSA at N = $N has R̂ = $rhat; this figure draws pooled bands only")
    model = ladder(:discrete, :multbin, true)
    thin = max(1, div(size(ch, 1) * size(ch, 3), NDRAW))
    sub = ch[1:thin:end, :, :]
    Random.seed!(7)                              # as s2_9_trajectories.jl: r drawn from its conditional posterior
    rd = vec([sample_index(l) for l in returned(model, sub)])
    return rd, vec(sub[:R0]), vec(sub[:mean_ip]), vec(sub[:ρ])
end

quant(M, p) = [quantile(M[t, :], p) for t in axes(M, 1)]

panels = Dict{Tuple{Int,Int},Any}(); Tmax = Dict{Int,Int}()
csv = open(joinpath(RES, "s2_10_prevalence_bands.csv"), "w")
println(csv, "N,model,day,lo,med,hi,truth,observed")
for N_s in NS
    β_s, ρ_s = set_data!(N_s)
    T = nsteps; Tmax[N_s] = T
    _, prev_true = glct_traj(dph_nom.S, dph_nom.α, β_s, ρ_s, T)
    prev_obs = linelist_prev(th, delta, T)
    for (i, (rung, name, map, col)) in enumerate(MODELS)
        rd, R0, ip, ρv = draws(rung)
        prev = zeros(T + 1, length(rd))
        for j in eachindex(rd)
            prev[:, j] = (map === :ode ? ode_traj : map_traj)(rd[j], R0[j], ip[j], ρv[j], T)[2]
        end
        keep = [j for j in axes(prev, 2) if !any(isnan, prev[:, j])]
        length(keep) < length(rd) && @warn "$name, N = $N_s: $(length(rd) - length(keep)) ODE solves failed and were dropped"
        P = prev[:, keep] ./ N_s
        lo, md, hi = quant(P, 0.025), quant(P, 0.5), quant(P, 0.975)
        for k in 0:T
            @printf(csv, "%d,%s,%d,%.6g,%.6g,%.6g,%.6g,%.6g\n", N_s, name, k, lo[k+1], md[k+1], hi[k+1], prev_true[k+1] / N_s, prev_obs[k+1] / N_s)
        end
        panels[(i, N_s)] = (lo = lo, md = md, hi = hi, truth = prev_true ./ N_s, obs = prev_obs ./ N_s, col = col, name = name)
        @printf("N = %6d  %-20s  %4d draws; peak proportion infectious %.3f [%.3f, %.3f] (truth %.3f)\n", N_s, name, length(keep),
                quantile(vec(maximum(P, dims = 1)), 0.5), quantile(vec(maximum(P, dims = 1)), 0.025),
                quantile(vec(maximum(P, dims = 1)), 0.975), maximum(prev_true) / N_s)
    end
end
close(csv)

# ------------------------------------------------------------ figure
fmt(n) = n >= 1000 ? replace(string(n), r"(\d)(?=(\d{3})+$)" => s"\1,") : string(n)
sty = (tickfontsize = 7, guidefontsize = 8, titlefontsize = 9, legendfontsize = 7, framestyle = :box,
       grid = :y, gridalpha = 0.15, gridlinewidth = 0.5, foreground_color_legend = nothing, background_color_legend = nothing)
plts = []
for (i, (rung, name, map, col)) in enumerate(MODELS), N_s in NS
    d = panels[(i, N_s)]; T = Tmax[N_s]; xs = 0:T
    first_col = N_s == NS[1]; last_row = i == length(MODELS)
    p = plot(; xlims = (0, T), ylims = (0, 1), yticks = 0:0.25:1, sty...,
             title = i == 1 ? "N = $(fmt(N_s))" : "", xlabel = last_row ? "Day" : "",
             ylabel = first_col ? "$name\nproportion infectious" : "", legend = false)
    plot!(p, xs, d.md; ribbon = (d.md .- d.lo, d.hi .- d.md), fillalpha = 0.25, color = d.col, lw = 1.5, label = "Posterior median, 95% CrI")
    plot!(p, xs, d.truth; color = "#1f1e1c", lw = 1, label = "True mean-field trajectory")
    scatter!(p, xs, d.obs; ms = 1.8, msw = 0, color = "#8f8d87", label = "Observed (line list)")
    push!(plts, p)
end
plot!(plts[end]; legend = :topright)              # one legend, in the last panel, whose upper half is empty
fig = plot(plts..., layout = (length(MODELS), length(NS)), size = (680, 860), left_margin = 3Plots.mm, dpi = 300)
savefig(fig, joinpath(FIGS, "s2_fig_trajectory_panels.pdf"))
savefig(fig, joinpath(FIGS, "s2_fig_trajectory_panels.png"))
println("wrote figs/s2_fig_trajectory_panels.{pdf,png} and out/s2_10_prevalence_bands.csv")
