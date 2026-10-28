#!/bin/zsh
# Rerun the chain-binomial coverage grid, the term attribution and the fits to data drawn from the
# working model (supplement S4, interval calibration)
# into S7_OUT (default out), in resumable chunks: each Julia process is
# stopped at 600 s and the next chunk resumes from the CSVs. Inputs fixed in advance (seeds, replicate lists) are
# unchanged and every fit is seeded, so a rerun with unchanged code reproduces out/ exactly.
# Usage (from anywhere): [S7_OUT=<dir>] path/to/baseline_grid/rerun_all.sh
cd "$(dirname "$0")"
export S7_OUT=${S7_OUT:-out}
SD=.
mkdir -p $SD/$S7_OUT; LOG=$SD/$S7_OUT/rerun_all.log
capped() {  # capped <logfile> <cmd...>: run with a 600-s cap
  local lf=$1; shift; "$@" >> $lf 2>&1 & local P=$! t=0
  while kill -0 $P 2>/dev/null; do sleep 5; t=$((t+5)); (( t >= 600 )) && { kill $P; echo "   STOPPED at 10-min cap" >> $lf; break; }; done
  wait $P 2>/dev/null
}
say() { echo "[$(date +%T)] $*" >> $LOG; }
say "start; S7_OUT=$S7_OUT"
say "sim_check";  capped $SD/$S7_OUT/sim_check.log julia --project=@. -t 10 $SD/sim_check.jl
say "cb";  SIM=cb  MAXCHUNKS=12 $SD/run_chunks.sh
say "term attribution"
for i in $(seq 1 8); do
  capped $SD/$S7_OUT/final_step2_chunks.log env INIT=data julia --project=@. -t 10 $SD/final_step2_term_attribution.jl
  tail -2 $SD/$S7_OUT/final_step2_chunks.log | grep -q "remaining: 0$" && break
done
say "working-model data"
for i in $(seq 1 12); do
  capped $SD/$S7_OUT/final_step3_chunks.log julia --project=@. -t 10 $SD/final_step3_working_model.jl
  tail -2 $SD/$S7_OUT/final_step3_chunks.log | grep -q "remaining: 0$" && break
done
say "summaries"
capped $SD/$S7_OUT/summarise.log env SIM=cb julia --project=@. $SD/sim_summarise.jl
capped $SD/$S7_OUT/summarise.log julia --project=@. $SD/final_step2_summarise.jl
capped $SD/$S7_OUT/summarise.log julia --project=@. $SD/final_step3_summarise.jl
say "done"
