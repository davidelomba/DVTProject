#!/usr/bin/env bash
# Two myocarditis series of three runs each, one after the other, through
# run_myo_t05.sh: hints 7d4be67fec0f with the tags in config.py, then the same
# with qwen3.8:27b for Agents 1 and 2 and the verifier. Each series is scored
# and its agreement measured by run_myo_t05.sh.
#
#   myo/output_t05_np2048_v4_r1 .. _r3       config.py tags (qwen3.6:27b)
#   myo/output_t05_np2048_v4_q38_r1 .. _r3   qwen3.8:27b
#
# Usage, from the project root on mari:
#   mkdir -p logs && nohup bash run_myo_v4.sh > logs/run_myo_v4.log 2>&1 &

set -euo pipefail

cd "$(dirname "$0")"

RUN_TAG=t05_np2048_v4 bash run_myo_t05.sh
RUN_TAG=t05_np2048_v4_q38 MODEL=qwen3.8:27b bash run_myo_t05.sh
