# S6: our runs of the authors' SDS R package (KhudaBukhsh et al. 2020; github cbskust/SDS.Epidemic @ 3ecb29d), code
# unchanged, on exported S3/S4 datasets: SDS.MCMC with Gamma(0.001, 0.001) priors, RAM-adapted random-walk Metropolis, 6,000
# iterations (1,000 burn-in), two chains; fitn = FALSE (n known, never-infected rows in the data) and fitn = TRUE
# (n imputed from NegBin(K, τ_∞)). Estimates are converted to the simulator's scale: the SDS mean field is per
# susceptible (ρ_SDS = I0/S0, hazard β_SDS I/n), so β = β_SDS (1 + ρ_SDS), R0 = β/γ, mean IP = 1/γ, N = n + m.
# Usage (from the supplementary folder): Rscript S3_S6_dsa_comparison/s6_2_sds.R [cores]
# (the package is read from S3_S6_dsa_comparison/external/Rlib; get_authors_code.sh downloads and installs it there)
args <- commandArgs(trailingOnly = TRUE); cores <- if (length(args)) as.integer(args[1]) else 3
D <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])))   # folder of this script
.libPaths(c(file.path(D, "external", "Rlib"), .libPaths()))
suppressMessages({library(MASS); library(SDSMCMC); library(parallel)})
files <- list.files(file.path(D, "external", "data"), pattern = "^sds_.*\\.csv$", full.names = TRUE)
out <- file.path(D, "out", "s6_sds_R.csv")
burn <- 1000; nrep <- 6000
rhat2 <- function(a, b) {             # split-R̂ over two chains (four halves)
  h <- length(a) %/% 2; ch <- list(a[1:h], a[(h+1):(2*h)], b[1:h], b[(h+1):(2*h)])
  W <- mean(sapply(ch, var)); B <- h * var(sapply(ch, mean)); sqrt(((h - 1) / h * W + B / h) / W)
}
jobs <- expand.grid(f = files, fitn = c(FALSE, TRUE), stringsAsFactors = FALSE)
run <- function(i) {
  f <- jobs$f[i]; fitn <- jobs$fitn[i]
  d <- read.csv(f); Tm <- d$Tmax[1]; dat <- as.matrix(d[, c("inf", "rec")])
  m <- sum(dat[, 1] == 0); K <- sum(dat[, 1] > 0 & dat[, 1] < Tm)
  dl <- dat[, 2] - dat[, 1]; ok <- dat[, 1] < Tm & dat[, 2] < Tm
  ic <- c(0.3, 1 / mean(dl[ok]), m / 1000, 2 * K)
  set.seed(1000 * i + 1)
  t0 <- proc.time()[["elapsed"]]
  ch <- lapply(1:2, function(c) {
    capture.output(th <- SDS.MCMC(data = dat, Tmax = Tm, fitn = fitn, nrepeat = nrep,
                                  prior.a = c(0.001, 0.001, 0.001), prior.b = c(0.001, 0.001, 0.001),
                                  ic = ic * exp(0.1 * rnorm(4))))
    th[(burn + 1):nrep, , drop = FALSE]
  })
  el <- proc.time()[["elapsed"]] - t0
  conv <- function(th) {
    beta <- th[, 1] * (1 + th[, 3]); cbind(R0 = beta / th[, 2], mu = 1 / th[, 2], beta = beta,
                                          N = if (fitn) th[, 4] + m else rep(NA, nrow(th)))
  }
  a <- conv(ch[[1]]); b <- conv(ch[[2]]); all <- rbind(a, b)
  q <- function(v) c(median(v), quantile(v, c(0.025, 0.975)))
  data.frame(file = basename(f), fitn = fitn, K = K, Tmax = Tm, time_s = el,
             R0_med = q(all[, "R0"])[1], R0_lo = q(all[, "R0"])[2], R0_hi = q(all[, "R0"])[3],
             mu_med = q(all[, "mu"])[1], mu_lo = q(all[, "mu"])[2], mu_hi = q(all[, "mu"])[3],
             beta_med = q(all[, "beta"])[1], N_med = if (fitn) q(all[, "N"])[1] else NA,
             N_lo = if (fitn) q(all[, "N"])[2] else NA, N_hi = if (fitn) q(all[, "N"])[3] else NA,
             rhat_R0 = rhat2(a[, "R0"], b[, "R0"]), rhat_mu = rhat2(a[, "mu"], b[, "mu"]))
}
sel <- if (length(args) > 1) as.integer(strsplit(args[2], ",")[[1]]) else seq_len(nrow(jobs))
res <- do.call(rbind, mclapply(sel, run, mc.cores = cores))
write.csv(res, out, row.names = FALSE, append = FALSE)
print(res[, c("file", "fitn", "time_s", "R0_med", "R0_lo", "R0_hi", "mu_med", "N_med", "rhat_R0")])
