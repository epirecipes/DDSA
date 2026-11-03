# Discrete dynamic survival analysis of epidemic case line lists: code and analyses

This repository contains the code, notebooks, and supplementary analyses used for the manuscript  *Discrete dynamic survival analysis of epidemic case line lists*. It is archived on Zenodo and records the manuscript's analyses.

To apply discrete dynamic survival analysis (DDSA) to your own line lists, use the Julia package
[DiscreteDynamicSurvivalAnalysis.jl](https://github.com/epirecipes/DiscreteDynamicSurvivalAnalysis.jl). Its vignettes reproduce the analyses below.

We are finalising our manuscript. A link to the preprint will be added here ![](https://img.shields.io/badge/Preprint-Coming_Soon-blue)

## Synthetic data case scenarios

- [DDSA data-availability scenarios](Data_avail_GLCT_Boxcar_Geo/ddsa_data_avail.md): This notebook evaluates what each likelihood term contributes: a synthetic outbreak under three data-availability scenarios, with a geometric infectious period and a negative binomial one, fitted through the GLCT and Boxcar forward maps.
- [Comparison to ODE count models](DDSA_vs_ODE_count/ddsa_vs_ode_count.md): This notebook compares DDSA with ODE count models fitted to the same synthetic outbreak.

## Real-world case scenarios

- [H1N1 outbreak](H1N1_WSU/dsa_WSU_Turing.md): H1N1 outbreak at Washington State University (WSU) in 2009.
- Ebola outbreak ([wave 1](Ebola_DRC_2018/DSA_turing_SIRmap_W1.md), [wave 2](Ebola_DRC_2018/DSA_turing_SIRmap_W2.md), [wave 3](Ebola_DRC_2018/DSA_turing_SIRmap_W3.md)): The 2018–2020 Ebola virus disease (EVD) epidemic in the Democratic Republic of the Congo (DRC).
- [Measles outbreak](Measles_Hagelloch_1861/dsa_hagelloch_GLCTvsBoxcar.md): The 1861 measles outbreak in Hagelloch, Germany.

## Supplementary analyses

[`Supplementary_analyses/supplementary_analyses.md`](Supplementary_analyses/supplementary_analyses.md) is a guide to the analyses of the supplementary material. For each one, it describes what was tested and how, lists the scripts and the command that reruns them, and shows the results.

## Requirements and rerunning

- **Julia 1.11.6.** The environment is `Project.toml` with `Manifest.toml` in the root folder:

  ```sh
  julia --project=. -e 'using Pkg; Pkg.instantiate()'
  ```

- **Quarto** with its Julia engine. Each analysis notebook uses the root environment and 10 Julia
  threads. To render only the Markdown version:

  ```sh
  quarto render H1N1_WSU/dsa_WSU_Turing.qmd --to gfm
  ```

- **R and Python**, for the comparison with other published continuous-time DSA code in
  `Supplementary_analyses/S3_S6_dsa_comparison/`. The versions used are listed in
  `sds_r_versions.txt` and `dsa_refactor_requirements.txt`.

## Data

| File | Source |
|:--|:--|
| `H1N1_WSU/WSU.csv` | `WSU.csv` in the SDS.Epidemic repository of KhudaBukhsh et al., <https://github.com/cbskust/SDS.Epidemic>. The infection times were reconstructed from daily case counts and the recovery times simulated from an exponential distribution with rate 1/5.5 per day. |
| `Ebola_DRC_2018/EVD_Wave{1,2,3}.csv` | "wasiur/DSAofEbola: Dynamic Survival Analysis (DSA)", v.2.0.0, doi [10.5281/zenodo.6104188](https://doi.org/10.5281/zenodo.6104188) |
| `Measles_Hagelloch_1861/hagelloch.csv` | the R package `outbreaks`, data set `measles_hagelloch_1861` (<https://CRAN.R-project.org/package=outbreaks>), formatted from `hagelloch.df` in the R package `surveillance`; collected by Pfeilsticker (1863) and Oesterle (1992). Four columns were added for the manuscript: `t_infection`, `t_rash`, `inf_period` and `t_recovery`. |

## Citation

Please cite the manuscript:

> S. Montes-Olivas, S. Abbott, M. B. Gravenor, S. Funk and S. D. W. Frost.
> Discrete dynamic survival analysis of epidemic case line lists. In preparation.

`CITATION.cff` has the citation metadata.

## Licence

The code is released under the MIT licence (`LICENSE`). The data files come from the sources
listed above.