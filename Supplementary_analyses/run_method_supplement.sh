#!/bin/zsh
# Reproduce every analysis of the supplement (Sections S2–S4), then regenerate the summaries, LaTeX tables and Figure S2.
# Seeds are fixed in the scripts. Every Julia process runs from this folder (with the shared environment found by
# --project=@.) and with a time cap; a process that ignores the stop signal is force-killed 30 s later. Resumable steps
# (S3, S5, S7) run in chunks and pick up from their CSVs, so re-running the script continues an interrupted run.
#
# Usage (from this folder; the script can also be called by its path from anywhere):
#   ./run_method_supplement.sh                 all sections: S2 S3 S5 VALID S7 TABLES FIGURES (about a day)
#   ./run_method_supplement.sh S2 S7           only these sections
#   ONLY="S2_1 S2_5" ./run_method_supplement.sh S2   only S2 steps whose label starts with these
#   DRY=1 ./run_method_supplement.sh            print the commands without running them
#
# Approximate run times on 10 threads: S2 4.5–5 h plus about 6 h for the s2_8 grids (mostly the ODE, r ≤ 40); S3 2 h;
# S5 1 h 20 min; VALID about 3 h; S7 (baseline grid) under 2 h 30 min; TABLES seconds; FIGURES about 5 min.
# Script section names keep the numbering of the analyses: S3 and VALID feed supplement S3, S5 and S7 supplement S4.
# Logs: S2_identifiability/out/logs/ (S2), S3_S6_dsa_comparison/out/logs/, S7_stress_tests/out/logs/, tables.log.
# Figures S1–S2 and the tables (supp_tables/) are also copied to a manuscript folder if one exists at MANUSCRIPT_DIR
# (default ../Discrete_DSA_manuscript); otherwise the copies are skipped.
cd "$(dirname "$0")"
MS=${MANUSCRIPT_DIR:-../Discrete_DSA_manuscript}
SA=.; D2=$SA/S2_identifiability; D3=$SA/S3_S6_dsa_comparison; D7=$SA/S7_stress_tests; LOG=$D2/out/logs
mkdir -p $LOG $D3/out/logs $D7/out/logs
SECTIONS=($@); (( ${#SECTIONS} )) || SECTIONS=(S2 S3 S5 VALID S7 TABLES FIGURES)
ONLY=(${=ONLY})

run() {   # run <logfile> <cap seconds> <command...>: run one process with a cap; force-kill if it ignores SIGTERM
  local log=$1 cap=$2; shift 2
  if [[ -n $DRY ]]; then echo "   [dry] (cap ${cap}s) $*  >> $log"; return 0; fi
  "$@" >> $log 2>&1 &
  local pid=$! t=0
  while kill -0 $pid 2>/dev/null; do
    sleep 5; t=$((t+5))
    if (( t >= cap )); then
      kill $pid; echo "   STOPPED at the ${cap}-s cap" >> $log
      sleep 30; kill -0 $pid 2>/dev/null && { kill -9 $pid; echo "   force-killed (no exit after SIGTERM)" >> $log; }
      break
    fi
  done
  wait $pid 2>/dev/null
}

step() {  # step <label> <env assignments...> -- <script> [threads] [cap]: one capped Julia run (S2)
  local label=$1; shift
  local envs=(); while [[ $1 != "--" ]]; do envs+=$1; shift; done; shift
  local script=$1 threads=${2:-10} cap=${3:-600}
  if (( ${#ONLY} )); then local ok=0; for f in $ONLY; do [[ $label == $f* ]] && ok=1; done; (( ok )) || return; fi
  echo "[$(date +%T)] $label: ${envs[*]} $script"
  [[ -z $DRY ]] && : > $LOG/$label.log
  run $LOG/$label.log $cap env $envs julia --project=@. -t $threads $SA/$script
}

chunks() {   # chunks <logfile> <maxchunks> <cap> <script> [env...]: resumable runs until "remaining: 0"
  local log=$1 max=$2 cap=$3 script=$4; shift 4
  echo "[$(date +%T)] chunks: $script $*"
  for i in $(seq 1 $max); do
    [[ -z $DRY ]] && echo "[$(date +%T)] chunk $i $script" >> $log
    run $log $cap env "$@" julia --project=@. -t 10 $script
    [[ -n $DRY ]] && break
    grep -q "remaining: 0$" <(tail -3 $log) && { echo "[$(date +%T)] all jobs done" >> $log; break; }
  done
}

for s in $SECTIONS; do
  echo "==== $s"
  case $s in
  S2)   # identifiability ladder (synthetic outbreak, N = 99), attack-rate sweep, grid posteriors and the larger-population trajectories
    step S2_0_check            --                        S2_identifiability/s2_0_check_likelihoods.jl 4
    for r in 1 2 3 4 5 6 7 8; do
    step S2_1_rung$r           RUNGS=$r --               S2_identifiability/s2_1_rungs_2x2_unknownN.jl 4
    done
    step S2_1_2x2              GRID2X2=1 --              S2_identifiability/s2_1_rungs_2x2_unknownN.jl 4
    step S2_1_unknownN_500     NWIDE=500 --              S2_identifiability/s2_1_rungs_2x2_unknownN.jl 4
    step S2_1_unknownN_20090   NWIDE=20090 --            S2_identifiability/s2_1_rungs_2x2_unknownN.jl 4
    step S2_2_wide_123         TASK=wide WIDE_RUNGS=1,2,3 -- S2_identifiability/s2_2_corroborations.jl 4
    step S2_2_wide_456         TASK=wide WIDE_RUNGS=4,5,6 -- S2_identifiability/s2_2_corroborations.jl 4
    step S2_2_misc             TASK=misc --              S2_identifiability/s2_2_corroborations.jl 4
    step S2_2_orig             TASK=orig --              S2_identifiability/s2_2_corroborations.jl 4
    step S2_3_ridge_r          --                        S2_identifiability/s2_3_ridge_and_r.jl 4
    step S2_4_sweep_N99        SWEEP_N=99 --             S2_identifiability/s2_4_tau_sweep_mcmc.jl 4
    for c in 1.5 2.0 3.16; do
    step S2_4_sweep_N1000_12ch_R0$c SWEEP_N=1000 SWEEP_CHAINS=12 SWEEP_WARMUP=2000 SWEEP_SUFFIX=_N1000_12ch CELLS=$c -- S2_identifiability/s2_4_tau_sweep_mcmc.jl 10
    done
    step S2_5_grid_N99         GRID_N=99 --              S2_identifiability/s2_5_tau_sweep_grid.jl
    for c in 1.5 2.0 3.16; do
    step S2_5_grid_N1000_R0$c  GRID_N=1000 CELLS=$c --   S2_identifiability/s2_5_tau_sweep_grid.jl
    done
    step S2_6_unknownN_seed    --                        S2_identifiability/s2_6_unknownN_seedprior_grid.jl
    step S2_6_unknownN_fraction PRIOR=fraction --        S2_identifiability/s2_6_unknownN_seedprior_grid.jl
    step S2_6_unknownN_fraction_refine2 PRIOR=fraction REFINE=2 -- S2_identifiability/s2_6_unknownN_seedprior_grid.jl
    # trajectories of map–Poisson, ODE–Poisson, infection-only DDSA and full-data DDSA at N = 99, 1,000 and 10,000: the
    # first three from the numerical grid posteriors of s2_8 (run first), full-data DDSA fitted by s2_9, which also draws
    # the figures (see its header); a stopped s2_9 run keeps the chains already saved and resumes from them
    for m in map inf ode; do
      for n in 99 1000 10000; do
        step S2_8_grid_${m}_N$n  GRID_MODEL=$m GRID_N=$n --  S2_identifiability/s2_8_grid_poisson.jl 10 $([[ $m == ode ]] && echo 14400 || echo 1800)
      done
    done
    # refinement checks at N = 10,000: the map grid with μ steps halved and 98 × 98 windows; the ODE grid with μ steps
    # halved over the range holding the posterior mass
    step S2_8_grid_map_N10000_refine2 GRID_MODEL=map GRID_N=10000 REFINE=2 -- S2_identifiability/s2_8_grid_poisson.jl 10 3600
    step S2_8_grid_ode_N10000_refine2 GRID_MODEL=ode GRID_N=10000 REFINE=2 WIN_REFINE=1 MU_RANGE=6,13 -- S2_identifiability/s2_8_grid_poisson.jl 10 14400
    step S2_9_traj_N99         TRAJ_N=99 --              S2_identifiability/s2_9_trajectories.jl 4 3600
    step S2_9_traj_N1000       TRAJ_N=1000 --            S2_identifiability/s2_9_trajectories.jl 4 7200
    step S2_9_traj_N10000      TRAJ_N=10000 --           S2_identifiability/s2_9_trajectories.jl 4 12600
    # grid posteriors of the attack-rate sweep: grid vs Turing log-joint check, refinement check (REFINE=2) and the
    # grid-vs-MCMC figure at N = 1,000 (about 35 min)
    step S2_7_grid_vs_turing_check -- S2_identifiability/s2_7_grid_vs_turing_check.jl 4
    for c in 2.0 3.16; do
    step S2_5_grid_N99_R0${c}_refine2 GRID_N=99 CELLS=$c REFINE=2 -- S2_identifiability/s2_5_tau_sweep_grid.jl 10 1200
    done
    step S2_5_grid_N1000_fig    GRID_N=1000 --            S2_identifiability/s2_5_tau_sweep_grid.jl 10 1800
    [[ -z $DRY && -d $MS/figures && -f $D2/figs/s2_fig_ridge_R0_meanIP.pdf ]] && cp $D2/figs/s2_fig_ridge_R0_meanIP.pdf $MS/figures/figS1_ridge_R0_meanIP.pdf
    [[ -z $DRY && -d $MS/figures && -f $D2/figs/s2_fig_trajectories_N10000.pdf ]] && cp $D2/figs/s2_fig_trajectories_N10000.pdf $MS/figures/figS2_trajectories_N10000.pdf
    ;;
  S3)   # DDSA vs continuous-time DSA (3,150 Bayesian fits and 900 complete-data analyses)
    chunks $D3/out/logs/chunks_s3_main.log 60 600 $D3/s3_1_run.jl RUN=s3_main REPS=50 ;;
  S5)   # Gamma infectious period (180 fits). Cap 900 s: the slowest fits (continuous-time DSA on infection times only)
        # take 6–12 min, so under a 600-s cap the last of them could never finish.
    run $D3/out/logs/s5_0_checks_run.log 600 julia --project=@. $D3/s5_0_checks.jl   # numerical checks → out/logs/s5_0_checks.log
    chunks $D3/out/logs/chunks_s5_main.log 60 900 $D3/s5_nonexp.jl RUN=s5_main REPS=20 BUDGET=240
    run $D3/out/logs/s5_negbin_uncapped.log 60 python3 -I $D3/s5_negbin_uncapped.py ;;   # NegBin DPH law without the cap → out/s5_negbin_uncapped.md
    # The Boxcar step refinement (s5_boxcar_dt.jl, ≈ 65 min on 10 threads) is run separately; its usage line is in the script header.
  VALID)   # supplement S3: the published codes in the settings of their own examples against our implementations
           # (Table S9), and the WSU population-size step (Table S11); about 3 h, mostly the SDS R fits
    X=$D3/external; PY="$X/.venv/bin/python -I -W ignore"
    [[ -d $X/DSA_refactor && -d $X/Rlib/SDSMCMC && -x $X/.venv/bin/python ]] || run $D3/out/logs/get_authors_code.log 1800 zsh $D3/get_authors_code.sh
    run $D3/out/logs/s6_4_sds_own_settings.log 10800 Rscript $D3/s6_4_sds_own_settings.R 6
    run $D3/out/logs/s6_5_sds_own_settings.log 3600 julia --project=@. -t 4 $D3/s6_5_sds_own_settings.jl
    for sd in 1 2 3 4 5; do
      run $D3/out/logs/s6_6_dsa_own_settings.log 600 zsh -c "$PY $D3/s6_6_dsa_own_settings.py data $sd 2>/dev/null"
      run $D3/out/logs/s6_6_dsa_own_settings.log 600 zsh -c "$PY $D3/s6_6_dsa_own_settings.py ll $sd 800 2>/dev/null"
    done
    run $D3/out/logs/s6_6_dsa_own_settings.log 1800 zsh -c "$PY $D3/s6_6_dsa_own_settings.py ll 1 10000 2>/dev/null"
    [[ -z $DRY ]] && : > $D3/out/s6_6_dsa_own_fits.jsonl
    for sd in 1 2 3 4 5; do
      run $D3/out/s6_6_dsa_own_fits.jsonl 1800 zsh -c "$PY $D3/s6_6_dsa_own_settings.py fit $sd 800 2>/dev/null | tail -1"
    done
    run $D3/out/logs/s6_7_dsa_own_settings.log 5400 julia --project=@. -t 5 $D3/s6_7_dsa_own_settings.jl
    [[ -z $DRY ]] && : > $D3/out/s6_6_dsa_own_at.jsonl
    for sd in 1 2 3 4 5; do for g in 800 10000; do
      run $D3/out/s6_6_dsa_own_at.jsonl 600 zsh -c "$PY $D3/s6_6_dsa_own_settings.py at $sd $g 2>/dev/null | tail -1"
    done; done
    run $D3/out/logs/s6_8_wsu_nstep.log 3600 julia --project=@. -t 3 $D3/s6_8_wsu_nstep.jl ;;
  S7)   # interval calibration: chain-binomial baseline grid, term attribution and working-model data
    run $D7/out/logs/s7_baseline_grid.log 36000 zsh $D7/baseline_grid/rerun_all.sh ;;
  TABLES)   # the LaTeX tables of S2–S4 (supp_tables/), copied to $MS/supp_tables/ if that folder exists
    if [[ -n $DRY ]]; then echo "   [dry] python3 -I $SA/make_supplement_tables.py"
    else python3 -I $SA/make_supplement_tables.py > $SA/tables.log 2>&1
      [[ -d $MS/supp_tables ]] && cp $SA/supp_tables/*.tex $MS/supp_tables/; fi
    ;;
  FIGURES)  # Figure S2 (proportion infectious) drawn from the saved S2 draws, copied to $MS/figures/ if that folder exists
    run $SA/figures.log 1800 julia --project=@. $D2/s2_10_trajectory_panels.jl
    [[ -z $DRY && -d $MS/figures ]] && cp $D2/figs/s2_fig_trajectory_panels.pdf $MS/figures/supp_s2_trajectory_panels.pdf ;;
  *) echo "unknown section $s (use S2 S3 S5 VALID S7 TABLES FIGURES)" ;;
  esac
done
echo "[$(date +%T)] done: ${SECTIONS[*]}"
