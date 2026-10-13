## Step 2: term attribution (chain binomial, r = 4, N = 1,000, 100 replicates per R₀; DPH model, final code, data-based starts)

| Variant | True R₀ | R₀ coverage | R₀ spread ratio | R₀ bias | β coverage | β spread ratio | β bias | R̂ > 1.01 | div > 0 |
|---|---|---|---|---|---|---|---|---|---|
| (a) Binomial final size | 1.3 | 0.28 | 6.02 | +0.011 | 0.37 | 3.99 | +0.012 | 0 | 2 |
| (a) Binomial final size | 2.0 | 0.51 | 2.71 | -0.005 | 0.50 | 2.61 | -0.003 | 0 | 2 |
| (a) Binomial final size | 3.0 | 0.65 | 1.95 | -0.001 | 0.69 | 1.95 | -0.002 | 0 | 5 |
| (b) no final-size term | 1.3 | 0.24 | 5.50 | +0.033 | 0.32 | 4.72 | +0.032 | 0 | 3 |
| (b) no final-size term | 2.0 | 0.50 | 2.83 | -0.003 | 0.44 | 2.85 | -0.001 | 0 | 4 |
| (b) no final-size term | 3.0 | 0.72 | 1.88 | -0.006 | 0.68 | 1.94 | -0.007 | 0 | 12 |
| (c) Normal, corrected variance | 1.3 | 0.28 | 5.44 | +0.025 | 0.33 | 4.54 | +0.026 | 0 | 3 |
| (c) Normal, corrected variance | 2.0 | 0.51 | 2.76 | -0.005 | 0.48 | 2.74 | -0.003 | 0 | 4 |
| (c) Normal, corrected variance | 3.0 | 0.69 | 1.90 | -0.002 | 0.71 | 1.93 | -0.004 | 0 | 4 |

### Sanity check of the final-size variance (not used for the decision)

| True R₀ | Empirical Var(K/N) over 100 replicates | Binomial τ(1−τ)/N | Corrected σ²_Z/N | ratio empirical / corrected |
|---|---|---|---|---|
| 1.3 | 4.61e-03 | 2.44e-04 | 4.21e-03 | 1.10 |
| 2.0 | 4.25e-04 | 1.62e-04 | 4.97e-04 | 0.85 |
| 3.0 | 8.68e-05 | 5.68e-05 | 8.90e-05 | 0.98 |
