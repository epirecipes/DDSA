#!/bin/zsh
# Run sim_run.jl in resumable chunks (each stops starting fits after BUDGET s and is killed at 600 s) until no jobs remain
# or MAXCHUNKS is reached.  Usage (from anywhere): SIM=cb REPS=10 MAXCHUNKS=4 path/to/baseline_grid/run_chunks.sh
cd "$(dirname "$0")"
D=.; mkdir -p $D/${S7_OUT:-out}; LOG=$D/${S7_OUT:-out}/chunks_${SIM:-cb}.log
for i in $(seq 1 ${MAXCHUNKS:-4}); do
  echo "[$(date +%T)] chunk $i" >> $LOG
  julia --project=@. -t 10 $D/sim_run.jl >> $LOG 2>&1 &
  P=$!; t=0
  while kill -0 $P 2>/dev/null; do sleep 5; t=$((t+5)); (( t >= 600 )) && { kill $P; echo "   STOPPED at 10-min cap" >> $LOG; break; }; done
  wait $P 2>/dev/null
  grep -q "remaining: 0$" <(tail -3 $LOG) && { echo "[$(date +%T)] all jobs done" >> $LOG; break; }
done
