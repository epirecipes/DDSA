# Export simulated datasets in the formats of the authors' software (S6).
#   SDS R package (KhudaBukhsh et al. 2020): two columns (infection time, recovery time); initial infectives have
#   infection time 0; never-infected individuals are rows with times 1e9 (so that SDS.MCMC(fitn = FALSE) counts n).
#   S3/S4 datasets, n = 1,000, R0 ∈ {1.5, 2, 4}, cut-off ∈ {peak, end}, replicates 1–5; the cut-off is in the file name.
#   DSA Python package (Di Lauro et al. 2022): S5 datasets, replicates 1–5: infection times (cases), infectious periods
#   (cases and seeds) and recovery times, one value per line.
# Usage (from the supplementary folder): julia --project=@. S3_S6_dsa_comparison/s6_0_export.jl
include(joinpath(@__DIR__, "s5_nonexp.jl"))
const XD = mkpath(joinpath(@__DIR__, "external", "data"))
MU = 5.0; RHO_S = 0.05
for R0 in (1.5, 2.0, 4.0), rep in 1:5
    n = 1000; m = round(Int, RHO_S * n)
    sim = sim_outbreak(Xoshiro(hash(("s3", n, R0, rep))), n, m, R0 / MU, Exponential(MU))
    mf = meanfield_cutoffs(R0, MU, m / (n + m))
    for (cut, T) in (("peak", ceil(mf.tpeak)), ("end", ceil(mf.t99)))
        open(joinpath(XD, "sds_n$(n)_R0$(R0)_$(cut)_rep$(rep).csv"), "w") do io
            println(io, "inf,rec,Tmax")
            for r in sim.seed_rec; println(io, "0.0,$r,$T"); end
            for (t, r) in zip(sim.inf_t, sim.rec_t); println(io, "$t,$r,$T"); end
            for _ in 1:(n - length(sim.inf_t)); println(io, "1e9,1e9,$T"); end
        end
    end
end
for rep in 1:5
    sim = simulate5(rep)
    open(io -> foreach(t -> println(io, t), sim.inf_t), joinpath(XD, "dsa5_rep$(rep)_infection_times.txt"), "w")
    open(io -> foreach(w -> println(io, w), vcat(sim.rec_t .- sim.inf_t, sim.seed_rec)), joinpath(XD, "dsa5_rep$(rep)_infectious_periods.txt"), "w")
    open(io -> foreach(t -> println(io, t), vcat(sim.rec_t, sim.seed_rec)), joinpath(XD, "dsa5_rep$(rep)_recovery_times.txt"), "w")
end
println("exported to $XD")
