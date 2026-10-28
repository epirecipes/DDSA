# Supplement S2 (unknown N; Table S7, last rows): rung 1 with N ~ U(91, 20090) summed out, numerical grid posterior,
# with the seed prior on one of two scales (PRIOR):
#   PRIOR=count (default)  I0 = ρN ~ Exp(1) (≈ ρ ~ Beta(1, 98) at N = 99). Reparameterised exactly: ρ flat on (0,1),
#                          and the joint prior of (ρ, N) is π_N(N) · π_I0(ρN) · N  (N = Jacobian of I0 = ρN).
#   PRIOR=fraction         ρ ~ Beta(1, 98) whatever N, the prior of the NUTS fit in s2_1 (whose chains split between
#                          two modes); the grid gives the posterior mass of each mode.
# In both, the forward map depends on ρ only, and N is summed out of the Binomial term (+ prior weight).
# REFINE=k divides the R0 and μ steps by k and multiplies the number of ρ points by k (grid-convergence check).
# Quadrature: midpoint rule; each cell is weighted by its width (dR, dμ, and ρ·dlog ρ on the log-ρ grid).
# Usage (from the supplementary folder): [PRIOR=fraction] [REFINE=2] julia --project=@. -t 10 S2_identifiability/s2_6_unknownN_seedprior_grid.jl
include(joinpath(@__DIR__, "s2_model.jl"))
using SpecialFunctions


const N_MAX = 20090                                   # same support as the wide-N fit: K..K+19999
const Ns    = collect(K:N_MAX)
const logC  = [loggamma(n + 1.0) - loggamma(K + 1.0) - loggamma(n - K + 1.0) for n in Ns]
const logNs = log.(Ns)
logπI0(x) = -x                                         # Exponential(1) on I0 = ρN
const PRIOR  = Symbol(get(ENV, "PRIOR", "count"))
PRIOR in (:count, :fraction) || error("PRIOR must be count or fraction")
const REFINE = parse(Int, get(ENV, "REFINE", "1"))
# N-dependent part of the prior weight in the N-sum: N π_I0(ρN) on the count scale, none on the fraction scale
nprior(i, ρ) = PRIOR === :count ? logNs[i] - ρ * Ns[i] : 0.0
logπρ(ρ) = PRIOR === :count ? 0.0 : logpdf(Beta(1.0, 98.0), ρ)   # ρ prior outside the N-sum (flat base on the count scale)

"""Value and gradient (∂τ, ∂ρ) of log Σ_N w(N, ρ) C(N,K) τ^K (1-τ)^(N-K) in Float64, with w = N π_I0(ρN) (count
scale) or 1 (fraction scale); uniform π_N is a constant. The gradient is a weighted average over the N posterior."""
function fs_val_grad(τ::Float64, ρ::Float64)
    # The summand is unimodal in N (log-concave), so stop once it is 40 log-units below the
    # running maximum after the peak: exact to double precision, and only tens of terms when τ is large.
    lt = K * log(τ); l1 = log1p(-τ)
    m = -Inf; S = 0.0; SN = 0.0
    @inbounds for i in eachindex(Ns)
        v = logC[i] + lt + (Ns[i] - K) * l1 + nprior(i, ρ)
        if v > m
            sc = exp(m - v); S = S * sc + 1.0; SN = SN * sc + Ns[i]; m = v
        else
            w = exp(v - m); S += w; SN += w * Ns[i]
            v < m - 40 && break
        end
    end
    EN = SN / S
    return m + log(S), K / τ - (EN - K) / (1 - τ), PRIOR === :count ? -EN : 0.0
end
function N_weights(τ, ρ)
    w = [logC[i] + K * log(τ) + (Ns[i] - K) * log1p(-τ) + nprior(i, ρ) for i in eachindex(Ns)]
    w .-= maximum(w); w = exp.(w); w ./ sum(w)
end

function ll_I0(r, R0, μ, ρ)
    r / μ < 1 || return -Inf
    μt, _ = mu_discrete(r, R0, μ, ρ)                   # τ and f do not depend on N (fractions)
    any(!isfinite, μt) && return -Inf
    f = safe_normalise(μt, nsteps)
    (all(isfinite, f) && abs(sum(f) - 1) < 0.01) || return -Inf
    τv = clamp(sum(μt) / N, 0.001, 0.999)              # N = 99 here only rescales μt back to fractions
    return fs_val_grad(τv, ρ)[1] + logπρ(ρ) + logpdf(Multinomial(K, f), th)
end

# Numerical grid posterior for rung 1 with N ~ U(91, 20090) summed out and the seed prior on the chosen scale; gives
# the posterior mass of the large-N / small-τ region (mode B).
# Coarse grid by default (a finer 290 × 180 × 72 grid did not finish in 18 min); REFINE=2 halves the R0 and μ steps.
# The N-sum depends on (τ, ρ) only, so it is tabulated once per ρ on a fine logit(τ) grid and
# interpolated; the interpolation error is checked against the exact sum below.
const OUT = open(joinpath(RES, (PRIOR === :count ? "s2_6_unknownN_seedprior_grid" : "s2_6_unknownN_fractionprior_grid") *
                              (REFINE == 1 ? "" : "_refine$REFINE") * ".txt"), "w")
say(x) = (println(x); println(OUT, x); flush(OUT))

dR = 0.05 / REFINE; R0s = collect(0.5 + dR / 2:dR:5.0 - dR / 2)
dμ = 0.25 / REFINE; μs  = collect(1.0 + dμ / 2:dμ:30.0 - dμ / 2)
lρs = collect(range(log(1e-6), log(0.3), length = 40REFINE)); ρs = exp.(lρs); dlρ = lρs[2] - lρs[1]
nR, nμ, nρ = length(R0s), length(μs), length(ρs)

# ---- tabulated N-sum: FS[i, iρ] = log Σ_N w(N, ρ) C(N,K) τ^K (1-τ)^(N-K) at τ = lgst(LT[i]); on the fraction scale
#      it does not depend on ρ, so one column is computed and copied
const LT = collect(range(log(0.001 / 0.999), log(0.999 / 0.001), length = 8001)); const dLT = LT[2] - LT[1]
lgst(x) = 1 / (1 + exp(-x))
FS = zeros(length(LT), nρ)
tfs = @elapsed Threads.@threads for i in eachindex(LT)
    if PRIOR === :count
        for iρ in 1:nρ; FS[i, iρ] = fs_val_grad(lgst(LT[i]), ρs[iρ])[1]; end
    else
        FS[i, :] .= fs_val_grad(lgst(LT[i]), ρs[1])[1]
    end
end
function fs_interp(τ, iρ)
    x = (log(τ / (1 - τ)) - LT[1]) / dLT; j = clamp(floor(Int, x) + 1, 1, length(LT) - 1); a = x - (j - 1)
    return (1 - a) * FS[j, iρ] + a * FS[j+1, iρ]
end
Random.seed!(11)
err = maximum(abs(fs_interp(τ, iρ) - fs_val_grad(τ, ρs[iρ])[1])
              for (τ, iρ) in zip(clamp.(rand(3000), 0.001, 0.999), rand(1:nρ, 3000)))
say("N-sum table $(length(LT)) τ × $nρ ρ in $(round(tfs, digits=1)) s; max |interp − exact| over 3000 random (τ, ρ) = $(round(err, sigdigits=2)) log-units")

lp = fill(-Inf, r_max, nμ, nR, nρ); τg = zeros(r_max, nμ, nR, nρ)
t = @elapsed Threads.@threads for iμ in 1:nμ
    for iR in 1:nR, iρ in 1:nρ, r in 1:r_max
        R0, μ, ρ = R0s[iR], μs[iμ], ρs[iρ]
        r / μ < 1 || continue
        μt, _ = mu_discrete(r, R0, μ, ρ)
        any(!isfinite, μt) && continue
        f = safe_normalise(μt, nsteps)
        (all(isfinite, f) && abs(sum(f) - 1) < 0.01) || continue
        τ = clamp(sum(μt) / N, 0.001, 0.999); τg[r, iμ, iR, iρ] = τ
        ll = fs_interp(τ, iρ) + logpdf(Multinomial(K, f), th)        # = ll_I0(r, R0, μ, ρ)
        # priors: R0, μ uniform (cell widths); ρ on a log grid (Jacobian ρ) with prior logπρ; r uniform
        isfinite(ll) && (lp[r, iμ, iR, iρ] = ll + logπρ(ρ) + log(dR) + log(dμ) + log(ρ * dlρ))
    end
end
# spot check against the exact ll_I0
Random.seed!(12); ci = CartesianIndices(lp)
chk = [(i = ci[rand(1:length(ci))]; isfinite(lp[i]) ? abs(lp[i] - log(dR) - log(dμ) - log(ρs[i[4]] * dlρ) -
       ll_I0(i[1], R0s[i[3]], μs[i[2]], ρs[i[4]])) : 0.0) for _ in 1:2000]
m = maximum(lp); w = exp.(lp .- m); Z = sum(w); w ./= Z
say("Seed prior: " * (PRIOR === :count ? "count scale, I0 = ρN ~ Exp(1)" : "fraction scale, ρ ~ Beta(1, 98)") * "; N ~ U($K, $N_MAX) summed out")
say("Grid $(r_max)×$nμ×$nR×$nρ (r, μ step $dμ, R0 step $dR, ρ ∈ [1e-6, 0.3] $nρ log points) in $(round(Int, t)) s; " *
    "max |grid ll − exact ll_I0| over 2000 random cells = $(round(maximum(chk), sigdigits=2))")
small = τg .< 0.5
say("Posterior mass with τ < 0.5 (large-N / small-attack region): $(sum(w[small]))")
subcrit = [R0s[i[3]] < 1 for i in CartesianIndices(w)]
say("Posterior mass with R0 < 1 (sub-critical; mode B of the NUTS fit lies here): $(sum(w[subcrit]))")
if sum(w[small]) > 0
    wsR = vec(sum(w .* small, dims = (1, 2, 4))); wsμ = vec(sum(w .* small, dims = (1, 3, 4)))
    say("   within τ < 0.5: R0 median $(round(R0s[searchsortedfirst(cumsum(wsR) ./ sum(wsR), 0.5)], digits=3)), mean IP median $(round(μs[searchsortedfirst(cumsum(wsμ) ./ sum(wsμ), 0.5)], digits=3))")
end
say("Max log posterior density: τ ≥ 0.5 region $(round(maximum(lp[.!small]), digits=2)); τ < 0.5 region $(round(maximum(lp[small .& isfinite.(lp)]), digits=2))")
iμB = argmin(abs.(μs .- 13.9)); iRB = argmin(abs.(R0s .- 0.52)); iρB = argmin(abs.(ρs .- 0.0094))
say("At mode B of the NUTS fit (R0≈0.52, μ≈13.9, ρ≈0.0094): max over r of log posterior = $(round(maximum(lp[:, iμB, iRB, iρB]), digits=2)) (global max $(round(m, digits=2)))")
lpm = dropdims(log.(sum(exp.(lp .- m), dims = 1)) .+ m, dims = 1)
locmax = Tuple{Float64,Float64,Float64,Float64}[]
for iμ in 2:nμ-1, iR in 1:nR, iρ in 2:nρ-1
    v = lpm[iμ, iR, iρ]; isfinite(v) || continue
    nb = [lpm[iμ+a, clamp(iR+b, 1, nR), iρ+c] for a in -1:1, b in -1:1, c in -1:1 if (a, b, c) != (0, 0, 0)]
    all(v .>= nb) && push!(locmax, (v, μs[iμ], R0s[iR], ρs[iρ]))
end
sort!(locmax, rev = true)
say("Local maxima of the r-summed log density (value, μ, R0, ρ), top 10:")
for x in locmax[1:min(10, end)]; say("   $(round.(x, sigdigits=4))"); end
wμ = vec(sum(w, dims = (1, 3, 4))); wRm = vec(sum(w, dims = (1, 2, 4))); wρm = vec(sum(w, dims = (1, 2, 3)))
wq(x, ww, p) = (o = sortperm(x); c = cumsum(ww[o]); x[o][searchsortedfirst(c, p)])
qw(x, ww) = "$(round(wq(x,ww,0.5),digits=3)) [$(round(wq(x,ww,0.025),digits=3)), $(round(wq(x,ww,0.975),digits=3))]"
Random.seed!(3); cells = CartesianIndices(w); cw = cumsum(vec(w))
idx = [cells[min(searchsortedfirst(cw, rand()), length(cw))] for _ in 1:4000]
Nd = Int[]; τd = Float64[]
for i in idx
    τ = τg[i]; ρ = ρs[i[4]]
    ww = N_weights(τ, ρ); u = rand(); c = 0.0; k = 1
    while k < length(ww) && (c += ww[k]) < u; k += 1; end
    push!(Nd, Ns[k]); push!(τd, τ)
end
I0d = Nd .* [ρs[i[4]] for i in idx]
say("Grid posterior: R0 = $(qw(R0s, wRm)) | meanIP = $(qw(μs, wμ)) | ρ = $(qw(ρs, wρm)) | τ = $(round(median(τd),digits=3)) [$(round(quantile(τd,0.025),digits=3)), $(round(quantile(τd,0.975),digits=3))]")
say("   N = $(median(Nd)) [$(quantile(Nd,0.025)), $(quantile(Nd,0.975))]  P(N ≤ 99) = $(round(mean(Nd .<= 99), digits=3)) | I0 = $(round(median(I0d),digits=2)) [$(round(quantile(I0d,0.025),digits=2)), $(round(quantile(I0d,0.975),digits=2))]")
say("   P(r) = $(round.(vec(sum(w, dims = (2, 3, 4))), digits=3)) | edge mass: R0 $(round(wRm[1] + wRm[end], sigdigits=2)), ρ $(round(wρm[1] + wρm[end], sigdigits=2))")
say("Reference, map–Poisson (Poisson counts, N = 99): R0 4.15 [2.42, 4.97], meanIP 8.84 [4.05, 12.23]")
close(OUT)
