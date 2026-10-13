# Summary table and figure for the S7 baseline grid: relative bias of the posterior median, 95% CrI coverage and
# relative CrI width for β, mean infectious period and R0, by cell and model; plus the conditioning (rejection) rates.
# Usage (from the supplementary folder): SIM=cb julia --project=@. S7_stress_tests/baseline_grid/sim_summarise.jl
ENV["GKSwstype"] = "100"
using Statistics, Printf, Plots
const SIM = get(ENV, "SIM", "cb")
const OUTDIR = joinpath(@__DIR__, get(ENV, "S7_OUT", "out"))
lines = collect(eachline(joinpath(OUTDIR, "results_$SIM.csv")))
hdr = split(lines[1], ","); col = Dict(h => i for (i, h) in enumerate(hdr))
rows = [split(l, ",") for l in lines[2:end]]
num(r, k) = parse(Float64, r[col[k]])
out = open(joinpath(OUTDIR, "summary_$SIM.md"), "w")
say(x) = (println(x); println(out, x))

cells = sort(unique([(Int(num(r, "N")), num(r, "R0"), Int(num(r, "r"))) for r in rows]))
say("# S7 baseline grid ($SIM): $(length(rows)) fit rows\n")
say("Conditioning: major outbreak, K ≥ 0.1 N. Rejection rate = 1 − 1/mean(attempts) per cell.\n")
say("| N | R0 | r | model | reps | rejection | β bias | β cover | β width | IP bias | IP cover | IP width | R0 bias | R0 cover | R0 width | R0 spread ratio | R̂ > 1.01 | div > 0 | median fit s |")
say("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
S = Dict()
for (N, R0, r) in cells, m in ("dph", "geo")
    rr = [x for x in rows if Int(num(x, "N")) == N && num(x, "R0") == R0 && Int(num(x, "r")) == r && x[col["model"]] == m]
    isempty(rr) && continue
    n = length(rr); rej = 1 - 1 / mean(num(x, "attempts") for x in rr)
    st = Dict()
    for (p, tr) in (("beta", "beta_true"), ("ip", "ip_true"), ("R0", "R0_true"))
        t = [num(x, tr) for x in rr]; med = [num(x, "$(p)_med") for x in rr]
        lo = [num(x, "$(p)_lo") for x in rr]; hi = [num(x, "$(p)_hi") for x in rr]
        # spread ratio: SD of the posterior medians across replicates / mean posterior SD (≈ CrI width / 3.92);
        # ≈ 1 when intervals are calibrated, > 1 when they are too narrow
        st[p] = (bias = mean((med .- t) ./ t), cover = mean((lo .<= t) .& (t .<= hi)), width = mean((hi .- lo) ./ t),
                 spread = std(med) / mean((hi .- lo) ./ 3.92))
    end
    S[(N, R0, r, m)] = (n = n, st = st)
    say(@sprintf("| %d | %.1f | %d | %s | %d | %.2f | %+.3f | %.2f | %.3f | %+.3f | %.2f | %.3f | %+.3f | %.2f | %.3f | %.2f | %d | %d | %.0f |",
        N, R0, r, m, n, rej, st["beta"].bias, st["beta"].cover, st["beta"].width, st["ip"].bias, st["ip"].cover, st["ip"].width,
        st["R0"].bias, st["R0"].cover, st["R0"].width, st["R0"].spread, count(num(x, "rhat") > 1.01 for x in rr), count(num(x, "div") > 0 for x in rr),
        median(num(x, "time_s") for x in rr)))
end
say("\nBias: mean relative error of the posterior median. Cover: share of replicates whose 95% CrI contains the truth (MC SE ≈ √(0.95·0.05/reps)). Width: mean CrI width / truth. Spread ratio: SD of posterior medians across replicates / mean posterior SD (≈ 1 if calibrated).")
close(out)

# figure: coverage vs N, by R0 (columns) and parameter (rows); lines = model × shape
pars = (("beta", "β"), ("ip", "Mean IP"), ("R0", "R₀")); R0s = sort(unique(c[2] for c in cells)); Ns = [100, 1000, 10_000]
sty = Dict(("dph", 4) => ("#2a78d6", :circle, "DPH, true r = 4"), ("geo", 4) => ("#eb6834", :utriangle, "Geometric, true r = 4"),
           ("dph", 1) => ("#1baf7a", :diamond, "DPH = geometric, true r = 1"))
P = []
for (p, plab) in pars, R0 in R0s
    pl = plot(xscale = :log10, xticks = (Ns, string.(Ns)), ylims = (0, 1.02), title = "$plab, R₀ = $R0", titlefontsize = 9,
              xlabel = p == "R0" ? "N" : "", ylabel = R0 == R0s[1] ? "95% CrI coverage" : "", legend = (p == "beta" && R0 == R0s[1]) ? :bottomleft : false,
              legendfontsize = 6, framestyle = :box, grid = true, gridalpha = 0.15)
    hline!(pl, [0.95], color = "#55534e", ls = :dash, label = false)
    for ((m, r), (c, mk, lab)) in sty
        xs = [N for N in Ns if haskey(S, (N, R0, r, m))]; isempty(xs) && continue
        plot!(pl, xs, [S[(N, R0, r, m)].st[p].cover for N in xs], color = c, marker = mk, lw = 1.5, ms = 4, label = lab)
    end
    push!(P, pl)
end
f = plot(P..., layout = (3, length(R0s)), size = (330 * length(R0s), 760), left_margin = 3Plots.mm, bottom_margin = 3Plots.mm, dpi = 200)
savefig(f, joinpath(OUTDIR, "coverage_$SIM.png")); savefig(f, joinpath(OUTDIR, "coverage_$SIM.pdf"))
