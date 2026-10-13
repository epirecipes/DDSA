# Supplement S2 (unknown N; Table S7, last row): rung 1 with N ~ U(91, 20090) summed out and a count-scale seed prior
# I0 = ρN ~ Exp(1). Exact grid posterior; tests whether the large-N, large-seed mode survives.
# Usage (from the supplementary folder): julia --project=@. -t 10 S2_identifiability/s2_6_unknownN_seedprior_grid.jl
# Wide-N rung with a count prior on the seed: I0 = ρN ~ Exponential(1) (≈ ρ ~ Beta(1, 98) at N = 99).
# Reparameterised exactly: sample ρ (flat on (0,1)); the joint prior of (ρ, N) is
#   π_N(N) · π_I0(ρN) · N        (N = Jacobian of I0 = ρN)
# so the forward map depends on ρ only and N is summed out of the Binomial term + prior weight.
include(joinpath(@__DIR__, "s2_model.jl"))
using SpecialFunctions


const N_MAX = 20090                                   # same support as the wide-N fit: K..K+19999
const Ns    = collect(K:N_MAX)
const logC  = [loggamma(n + 1.0) - loggamma(K + 1.0) - loggamma(n - K + 1.0) for n in Ns]
const logNs = log.(Ns)
logπI0(x) = -x                                         # Exponential(1) on I0 = ρN

"""Value and gradient (∂τ, ∂ρ) of log Σ_N N π_I0(ρN) C(N,K) τ^K (1-τ)^(N-K) in Float64
(uniform π_N is a constant). The gradient is a weighted average over the N posterior."""
function fs_val_grad(τ::Float64, ρ::Float64)
    # The summand is unimodal in N (log-concave), so stop once it is 40 log-units below the
    # running maximum after the peak: exact to double precision, and only tens of terms when τ is large.
    lt = K * log(τ); l1 = log1p(-τ)
    m = -Inf; S = 0.0; SN = 0.0
    @inbounds for i in eachindex(Ns)
        v = logC[i] + lt + (Ns[i] - K) * l1 + logNs[i] - ρ * Ns[i]
        if v > m
            sc = exp(m - v); S = S * sc + 1.0; SN = SN * sc + Ns[i]; m = v
        else
            w = exp(v - m); S += w; SN += w * Ns[i]
            v < m - 40 && break
        end
    end
    EN = SN / S
    return m + log(S), K / τ - (EN - K) / (1 - τ), -EN
end
function N_weights(τ, ρ)
    w = [logC[i] + K * log(τ) + (Ns[i] - K) * log1p(-τ) + logNs[i] + logπI0(ρ * Ns[i]) for i in eachindex(Ns)]
    w .-= maximum(w); w = exp.(w); w ./ sum(w)
end

function ll_I0(r, R0, μ, ρ)
    r / μ < 1 || return -Inf
    μt, _ = mu_discrete(r, R0, μ, ρ)                   # τ and f do not depend on N (fractions)
    any(!isfinite, μt) && return -Inf
    f = safe_normalise(μt, nsteps)
    (all(isfinite, f) && abs(sum(f) - 1) < 0.01) || return -Inf
    τv = clamp(sum(μt) / N, 0.001, 0.999)              # N = 99 here only rescales μt back to fractions
    return fs_val_grad(τv, ρ)[1] + logpdf(Multinomial(K, f), th)
end

# Exact grid posterior for rung 1 with N ~ U(91, 20090) summed out and the seed on the count scale,
# I0 = ρN ~ Exponential(1). Tests whether the large-N / small-τ mode (mode B) survives a count prior.
# Coarse grid (a finer 290 × 180 × 72 grid did not finish in 18 min).
# The N-sum depends on (τ, ρ) only, so it is tabulated once per ρ on a fine logit(τ) grid and
# interpolated; the interpolation error is checked against the exact sum below.
const OUT = open(joinpath(RES, "s2_6_unknownN_seedprior_grid.txt"), "w")
say(x) = (println(x); println(OUT, x); flush(OUT))

R0s = collect(0.525:0.05:4.975);     dR = 0.05
μs  = collect(1.125:0.25:29.875);    dμ = 0.25
lρs = collect(range(log(1e-6), log(0.3), length = 40)); ρs = exp.(lρs); dlρ = lρs[2] - lρs[1]
nR, nμ, nρ = length(R0s), length(μs), length(ρs)

# ---- tabulated N-sum: FS[i, iρ] = log Σ_N N π_I0(ρN) C(N,K) τ^K (1-τ)^(N-K) at τ = lgst(LT[i])
const LT = collect(range(log(0.001 / 0.999), log(0.999 / 0.001), length = 8001)); const dLT = LT[2] - LT[1]
lgst(x) = 1 / (1 + exp(-x))
FS = zeros(length(LT), nρ)
tfs = @elapsed Threads.@threads for i in eachindex(LT)
    for iρ in 1:nρ; FS[i, iρ] = fs_val_grad(lgst(LT[i]), ρs[iρ])[1]; end
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
        # priors: R0, μ uniform (cell widths); ρ flat base on (0,1) on a log grid (Jacobian ρ); r uniform
        isfinite(ll) && (lp[r, iμ, iR, iρ] = ll + log(dR) + log(dμ) + log(ρ * dlρ))
    end
end
# spot check against the exact ll_I0
Random.seed!(12); ci = CartesianIndices(lp)
chk = [(i = ci[rand(1:length(ci))]; isfinite(lp[i]) ? abs(lp[i] - log(dR) - log(dμ) - log(ρs[i[4]] * dlρ) -
       ll_I0(i[1], R0s[i[3]], μs[i[2]], ρs[i[4]])) : 0.0) for _ in 1:2000]
m = maximum(lp); w = exp.(lp .- m); Z = sum(w); w ./= Z
say("Grid $(r_max)×$nμ×$nR×$nρ (r, μ step $dμ, R0 step $dR, ρ ∈ [1e-6, 0.3] $nρ log points) in $(round(Int, t)) s; " *
    "max |grid ll − exact ll_I0| over 2000 random cells = $(round(maximum(chk), sigdigits=2))")
small = τg .< 0.5
say("Posterior mass with τ < 0.5 (large-N / small-attack region): $(sum(w[small]))")
say("Max log posterior density: τ ≥ 0.5 region $(round(maximum(lp[.!small]), digits=2)); τ < 0.5 region $(round(maximum(lp[small .& isfinite.(lp)]), digits=2))")
iμB = argmin(abs.(μs .- 13.9)); iRB = argmin(abs.(R0s .- 0.52)); iρB = argmin(abs.(ρs .- 0.0094))
say("At old mode B (R0≈0.52, μ≈13.9, ρ≈0.0094): max over r of log posterior = $(round(maximum(lp[:, iμB, iRB, iρB]), digits=2)) (global max $(round(m, digits=2)))")
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
say("Exact posterior: R0 = $(qw(R0s, wRm)) | meanIP = $(qw(μs, wμ)) | ρ = $(qw(ρs, wρm)) | τ = $(round(median(τd),digits=3)) [$(round(quantile(τd,0.025),digits=3)), $(round(quantile(τd,0.975),digits=3))]")
say("   N = $(median(Nd)) [$(quantile(Nd,0.025)), $(quantile(Nd,0.975))]  P(N ≤ 99) = $(round(mean(Nd .<= 99), digits=3)) | I0 = $(round(median(I0d),digits=2)) [$(round(quantile(I0d,0.025),digits=2)), $(round(quantile(I0d,0.975),digits=2))]")
say("   P(r) = $(round.(vec(sum(w, dims = (2, 3, 4))), digits=3)) | edge mass: R0 $(round(wRm[1] + wRm[end], sigdigits=2)), ρ $(round(wρm[1] + wρm[end], sigdigits=2))")
say("Reference, rung 2 (Poisson counts, N = 99): R0 4.15 [2.42, 4.97], meanIP 8.84 [4.05, 12.23]")
close(OUT)
