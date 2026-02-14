#!/bin/bash
set -e

# Create workspace directory
mkdir -p output/workspace

echo "=== Step 0: Preparing Input Data ==="
# Convert individual JSON files from instances/ to output/workspace/all_instances.jsonl
if [ -d "instances" ]; then
    echo "Converting instances/ to output/workspace/all_instances.jsonl..."
    python utils/dir_to_jsonl.py "instances/aime2025_scored/instances" "instances/aime2025_scored.jsonl"
else
    echo "Error: Directory instances/ not found."
    exit 1
fi

echo "=== Step 1: Scoring Instances (Pass@K) ==="
# Evaluate instances with 4 rollouts and 8 threads
python utils/filter_aime.py \
    --input_file "instances/aime2024.jsonl" \
    --output_file "instances/aime2024_qwen2_7b.jsonl" \
    --rollouts 30 \
    --threads 256

echo "=== Step 2: Selecting Hard Instances ==="
# Filter for pass_rate <= 0.2
python utils/filter_scored_instances.py \
    --input_file "instances/aime2025_scored.jsonl" \
    --output_file "instances/aime2025_filtered.jsonl" \
    --max_pass_rate 0.2

COUNT=$(wc -l < "output/workspace/hard_instances.jsonl")
echo "Selected $COUNT hard instances."

if [ "$COUNT" -eq 0 ]; then
    echo "No hard instances found. Exiting."
    exit 0
fi

echo "=== Step 3: Running Evolution Loop ==="
echo "Results will be saved to output/evolve_results"
python SE_Perf/perf_run.py \
    --config "configs/aime_evolve.yaml" \
    --input_file "instances/aime2024.jsonl" \
    --workers 128

echo "Pipeline Completed Successfully!"
