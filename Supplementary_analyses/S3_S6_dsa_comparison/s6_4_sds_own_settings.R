# S6: the authors' SDS R package (KhudaBukhsh et al. 2020; github cbskust/SDS.Epidemic @ 3ecb29d), code unchanged, in the
# settings of its own example scripts, for comparison with our re-implementation (s6_5_sds_own_settings.jl):
#   sellke  Example2.R: outbreaks from their Sellke() with n = 1,000, beta = 1.1, gamma = 0.8, rho = 0.05, T = 30
#           (REPS replicates), fitted with fitn = FALSE (n known) and fitn = TRUE (n imputed, Algorithm 4.1)
#   wsu     example1.R: WSU.csv, T = 104, fitn = TRUE, ic = (0.25, 0.2, 0.01, 1000)
# Priors Gamma(0.001, 0.001) on beta, gamma and rho, as in both examples. Four chains of BURN + KEEP iterations (their
# examples use one chain of 1,000 + 5,000 or 1,000 + 2,000), initial values jittered by exp(0.1 z).
# Estimates are converted to the simulator's scale as in s6_2_sds.R: beta = beta_SDS (1 + rho_SDS), R0 = beta/gamma,
# mean IP = 1/gamma, N = n + m. The script also writes their log-likelihood (llikelihood after SIR.ODE, dt = 0.01) at
# 20 parameter points per dataset, for the pointwise check.
# Usage (from the supplementary folder): Rscript S3_S6_dsa_comparison/s6_4_sds_own_settings.R [cores]
#   REPS=10 BURN=2000 KEEP=8000; EXT=<folder> reads the package from <folder>/Rlib (default S3_S6_dsa_comparison/external)
args <- commandArgs(trailingOnly = TRUE); cores <- if (length(args)) as.integer(args[1]) else 8
D <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])))
EXT <- Sys.getenv("EXT", file.path(D, "external"))
.libPaths(c(file.path(EXT, "Rlib"), .libPaths()))
suppressMessages({library(MASS); library(smfsb); library(ramcmc); library(plyr); library(SDSMCMC); library(parallel)})
REPS <- as.integer(Sys.getenv("REPS", "10")); BURN <- as.integer(Sys.getenv("BURN", "2000"))
KEEP <- as.integer(Sys.getenv("KEEP", "8000")); NCH <- 4
DD <- file.path(D, "external", "data"); dir.create(DD, showWarnings = FALSE, recursive = TRUE)
OUT <- file.path(D, "out")
ns <- asNamespace("SDSMCMC")

# ------------------------------------------------------------------ datasets
sets <- list()
for (rep in seq_len(REPS)) {
  f <- file.path(DD, sprintf("sdsown_sellke_rep%d.csv", rep))
  if (!file.exists(f)) {
    set.seed(20261027 + rep)
    pd <- Sellke(n = 1000, rho = 0.05, beta = 1.1, gamma = 0.8, Tmax = 30)
    write.csv(data.frame(inf = pd[, 1], rec = pd[, 2], Tmax = 30), f, row.names = FALSE)
  }
  sets[[length(sets) + 1]] <- list(name = sprintf("sellke_rep%d", rep), file = f, Tm = 30, fitn = c(FALSE, TRUE),
                                   ic = c(1.1, 0.8, 0.05, 1000), n_ll = 1000, pt = c(1.1, 0.8, 0.05))
}
wf <- file.path(DD, "sdsown_wsu.csv")
if (!file.exists(wf)) {
  w <- read.csv(file.path(D, "..", "..", "H1N1_WSU", "WSU.csv"))
  write.csv(data.frame(inf = w$V1, rec = w$V2, Tmax = 104), wf, row.names = FALSE)
}
sets[[length(sets) + 1]] <- list(name = "wsu", file = wf, Tm = 104, fitn = TRUE, ic = c(0.25, 0.2, 0.01, 1000),
                                 n_ll = 7803, pt = c(0.205, 0.195, 0.036))

# ------------------------------------------------------------------ pointwise log-likelihoods (their functions)
ll_rows <- list()
for (s in sets) {
  d <- read.csv(s$file); dat <- as.matrix(d[, c("inf", "rec")]); Tm <- s$Tm
  delta <- c(subset(dat[, 2] - dat[, 1], dat[, 1] < Tm & dat[, 2] < Tm), subset(Tm - dat[, 1], dat[, 1] < Tm & dat[, 2] >= Tm))
  nz <- sum(dat[, 1] < Tm & dat[, 2] < Tm)
  ti <- subset(dat[, 1], dat[, 1] > 0 & dat[, 1] < Tm)
  set.seed(7)
  for (j in 1:20) {
    p <- s$pt * exp(0.1 * rnorm(3))
    si <- ns$SIR.ODE(indata = ti, Tmax = Tm, beta = p[1], gamma = p[2], ic = c(1, p[3], 0))
    v <- ns$llikelihood(SI_ti = si, p.m = p, delta.t = delta, n.num = s$n_ll, nz.num = nz)
    ll_rows[[length(ll_rows) + 1]] <- data.frame(set = s$name, point = j, beta_sds = p[1], gamma = p[2], rho_sds = p[3],
                                                 n = s$n_ll, ll_sds = v)
  }
}
write.csv(do.call(rbind, ll_rows), file.path(OUT, "s6_4_sds_own_ll.csv"), row.names = FALSE)
cat("pointwise log-likelihoods written\n")

# ------------------------------------------------------------------ fits
rhat <- function(m) {                  # split-R̂ over the columns of a draws × chains matrix
  h <- nrow(m) %/% 2; ch <- cbind(m[1:h, , drop = FALSE], m[(h + 1):(2 * h), , drop = FALSE])
  W <- mean(apply(ch, 2, var)); B <- h * var(colMeans(ch)); sqrt(((h - 1) / h * W + B / h) / W)
}
jobs <- do.call(rbind, lapply(seq_along(sets), function(i) expand.grid(set = i, fitn = sets[[i]]$fitn, chain = 1:NCH)))
run <- function(k) {
  s <- sets[[jobs$set[k]]]; fitn <- jobs$fitn[k]
  dat <- as.matrix(read.csv(s$file)[, c("inf", "rec")])
  set.seed(1000 * k + 17)
  t0 <- proc.time()[["elapsed"]]
  capture.output(th <- SDS.MCMC(data = dat, Tmax = s$Tm, fitn = fitn, nrepeat = BURN + KEEP,
                                prior.a = c(0.001, 0.001, 0.001), prior.b = c(0.001, 0.001, 0.001),
                                ic = s$ic * c(exp(0.1 * rnorm(3)), 1)))
  list(th = th[(BURN + 1):(BURN + KEEP), , drop = FALSE], m = sum(dat[, 1] == 0), time = proc.time()[["elapsed"]] - t0)
}
res <- mclapply(seq_len(nrow(jobs)), run, mc.cores = cores)
rows <- list()
for (g in unique(paste(jobs$set, jobs$fitn))) {
  idx <- which(paste(jobs$set, jobs$fitn) == g); s <- sets[[jobs$set[idx[1]]]]; fitn <- jobs$fitn[idx[1]]
  conv <- function(r) { th <- r$th; beta <- th[, 1] * (1 + th[, 3])
    cbind(R0 = beta / th[, 2], mu = 1 / th[, 2], beta = beta, rho = th[, 3] / (1 + th[, 3]),
          N = if (fitn) th[, 4] + r$m else NA) }
  cs <- lapply(res[idx], conv)
  q <- function(v) c(median(v), quantile(v, c(0.025, 0.975)))
  row <- data.frame(set = s$name, fitn = fitn, impl = "SDS R", time_s = sum(sapply(res[idx], `[[`, "time")))
  for (v in c("R0", "mu", "beta", "rho", "N")) {
    if (v == "N" && !fitn) { row[paste0(v, c("_med", "_lo", "_hi"))] <- NA; next }
    a <- unlist(lapply(cs, function(x) x[, v])); row[paste0(v, c("_med", "_lo", "_hi"))] <- q(a)
  }
  row$rhat_R0 <- rhat(sapply(cs, function(x) x[, "R0"])); row$rhat_mu <- rhat(sapply(cs, function(x) x[, "mu"]))
  row$rhat_N <- if (fitn) rhat(sapply(cs, function(x) x[, "N"])) else NA
  rows[[length(rows) + 1]] <- row
}
out <- do.call(rbind, rows)
write.csv(out, file.path(OUT, "s6_4_sds_own_R.csv"), row.names = FALSE)
print(out[, c("set", "fitn", "R0_med", "R0_lo", "R0_hi", "mu_med", "N_med", "rhat_R0", "rhat_N")])
