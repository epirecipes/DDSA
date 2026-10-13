#!/bin/zsh
# Reproduce every analysis of the supplement (Sections S2–S8), then regenerate the summaries and LaTeX tables.
# Seeds are fixed in the scripts. Every Julia process runs from this folder (with the shared environment found by
# --project=@.) and with a time cap; a process that ignores the stop signal is force-killed 30 s later. Resumable steps
# (S3, S4, S5, S7) run in chunks and pick up from their CSVs, so re-running the script continues an interrupted run.
#
# Usage (from this folder; the script can also be called by its path from anywhere):
#   ./run_method_supplement.sh                 all sections: S2 S3 S4 S5 S6 S7 S8 TABLES (many hours)
#   ./run_method_supplement.sh S2 S8           only these sections
#   ONLY="S2_1 S2_5" ./run_method_supplement.sh S2   only S2 steps whose label starts with these
#   DRY=1 ./run_method_supplement.sh            print the commands without running them
#
# Approximate run times on 10 threads: S2 1.5–2 h; S3 2 h; S4 40 min; S5 1 h 20 min; S6 1 h 30 min (authors' code and
# benchmarks; run S6 on an otherwise idle machine, it times the fits); S7 1 h 30 min; S8 35 min; TABLES seconds.
# Logs: S2_identifiability/out/logs/ (S2, S8), S3_S6_dsa_comparison/out/logs/, S7_stress_tests/out/logs/, tables.log.
# Figures S1–S2 and the tables (supp_tables/) are also copied to a manuscript folder if one exists at MANUSCRIPT_DIR
# (default ../Discrete_DSA_manuscript); otherwise the copies are skipped.
cd "$(dirname "$0")"
MS=${MANUSCRIPT_DIR:-../Discrete_DSA_manuscript}
SA=.; D2=$SA/S2_identifiability; D3=$SA/S3_S6_dsa_comparison; D7=$SA/S7_stress_tests; LOG=$D2/out/logs
mkdir -p $LOG $D3/out/logs $D7/out/logs
SECTIONS=($@); (( ${#SECTIONS} )) || SECTIONS=(S2 S3 S4 S5 S6 S7 S8 TABLES)
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

step() {  # step <label> <env assignments...> -- <script> [threads] [cap]: one capped Julia run (S2, S8)
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
  S2)   # identifiability ladder (synthetic outbreak, N = 99) and the attack-rate sweep used by S8
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
    [[ -z $DRY && -d $MS/figures && -f $D2/figs/s2_fig_ridge_R0_meanIP.pdf ]] && cp $D2/figs/s2_fig_ridge_R0_meanIP.pdf $MS/figures/figS1_ridge_R0_meanIP.pdf
    ;;
  S3)   # DDSA vs continuous-time DSA (3,150 Bayesian fits and 900 complete-data analyses)
    chunks $D3/out/logs/chunks_s3_main.log 60 600 $D3/s3_1_run.jl RUN=s3_main REPS=50 ;;
  S4)   # unknown population size (540 fits; 10 replicates, N_max = 5n, set in s4_popsize.jl)
    chunks $D3/out/logs/chunks_s4_main.log 80 600 $D3/s4_popsize.jl RUN=s4_main S4_REPS=10 ;;
  S5)   # Gamma infectious period (180 fits). Cap 900 s: the slowest fits (continuous-time DSA on infection times only)
        # take 6–12 min, so under a 600-s cap the last of them could never finish.
    run $D3/out/logs/s5_0_checks_run.log 600 julia --project=@. $D3/s5_0_checks.jl   # numerical checks → out/logs/s5_0_checks.log
    chunks $D3/out/logs/chunks_s5_main.log 60 900 $D3/s5_nonexp.jl RUN=s5_main REPS=20 BUDGET=240 ;;
    # The Boxcar step refinement is run separately (≈ 65 min on 10 threads; output out/s5_boxcar_dt.csv):
    #   DTS=0.5,0.25 REPS=20 SCEN=A BUDGET=30000 julia --project=@. -t 10 S3_S6_dsa_comparison/s5_boxcar_dt.jl
  S6)   # computational cost and the authors' software (run on an otherwise idle machine)
    X=$D3/external
    # the authors' code is not distributed with this repository: download it at the commits we ran if it is missing
    [[ -d $X/DSA_refactor && -d $X/Rlib/SDSMCMC && -x $X/.venv/bin/python ]] || run $D3/out/logs/get_authors_code.log 1800 zsh $D3/get_authors_code.sh
    run $D3/out/logs/s6_0_export.log 600 julia --project=@. $D3/s6_0_export.jl
    run $D3/out/logs/s6_2_sds.log 5400 Rscript $D3/s6_2_sds.R 3
    [[ -z $DRY ]] && : > $D3/out/s6_dsa_python.jsonl
    for rep in 1 2 3 4 5; do for sc in A B; do
      # keep only the final JSON line of each fit (the authors' code prints progress and warnings)
      run $D3/out/s6_dsa_python.jsonl 900 zsh -c "PYTHONPATH=$X/shim:$X/DSA_refactor $X/.venv/bin/python -W ignore $D3/s6_3_dsa_python.py $rep $sc 100 2>/dev/null | tail -1"
    done; done
    run $D3/out/logs/s6_1_benchmarks.log 3600 julia --project=@. -t 1 $D3/s6_1_benchmarks.jl
    ;;
  S7)   # stress tests: baseline chain-binomial and Gillespie grids and term attribution, then one factor at a time
    run $D7/out/logs/s7_baseline_grid.log 36000 zsh $D7/baseline_grid/rerun_all.sh
    chunks $D7/out/logs/chunks_stress_main.log 60 600 $D7/stress_run.jl RUN=stress_main REPS=50 ;;
  S8)   # grid posteriors: grid vs Turing log-joint check, refinement check (REFINE=2) and the Figure S2 grid (N = 1,000, about 15 min)
    step S8_grid_vs_turing_check -- S2_identifiability/s2_7_grid_vs_turing_check.jl 4
    for c in 2.0 3.16; do
    step S8_grid_N99_R0${c}_refine2 GRID_N=99 CELLS=$c REFINE=2 -- S2_identifiability/s2_5_tau_sweep_grid.jl 10 1200
    done
    step S8_figS3_grid_N1000   GRID_N=1000 --            S2_identifiability/s2_5_tau_sweep_grid.jl 10 1800
    [[ -z $DRY && -d $MS/figures && -f $D2/figs/s5_fig_meanIP_exact_vs_mcmc_N1000.pdf ]] && cp $D2/figs/s5_fig_meanIP_exact_vs_mcmc_N1000.pdf $MS/figures/figS3_meanIP_exact_vs_mcmc_N1000.pdf
    ;;
  TABLES)   # the LaTeX tables of S3–S7 (supp_tables/), copied to $MS/supp_tables/ if that folder exists
    if [[ -n $DRY ]]; then echo "   [dry] python3 -I $SA/make_supplement_tables.py"
    else python3 -I $SA/make_supplement_tables.py > $SA/tables.log 2>&1
      [[ -d $MS/supp_tables ]] && cp $SA/supp_tables/*.tex $MS/supp_tables/; fi
    ;;
  *) echo "unknown section $s (use S2 S3 S4 S5 S6 S7 S8 TABLES)" ;;
  esac
done
echo "[$(date +%T)] done: ${SECTIONS[*]}"
