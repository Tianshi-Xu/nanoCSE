#!/bin/bash

# Configuration
CONFIG="configs/aime_evolve.yaml"
INPUT_DIR="output/aime_hard"           # Directory containing hard instances (from filter pipeline)
OUTPUT_BASE="output/evolve_results"    # Base directory for results
LOG_DIR="logs"

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_BASE"

echo "Using config: $CONFIG"
echo "Input directory: $INPUT_DIR"
echo "Output base: $OUTPUT_BASE"

# Find all JSON files in the input directory
# We can use `find` to get the list
FILES=($(find "$INPUT_DIR" -name "*.json" | sort))
NUM_FILES=${#FILES[@]}

if [ "$NUM_FILES" -eq 0 ]; then
    echo "No JSON files found in $INPUT_DIR"
    exit 1
fi

echo "Found $NUM_FILES instances to process."

# Set parallelism (number of concurrent instances to evolve)
MAX_JOBS=8
echo "Running up to $MAX_JOBS instances in parallel..."

# Function to process a single file (to be exported for parallel or used in loop)
process_file() {
    instance_path=$1
    config=$2
    output_base=$3
    job_id=$4

    # Extract instance ID from filename (e.g., "1873_A.json" -> "1873_A")
    filename=$(basename "$instance_path")
    instance_id="${filename%.*}"
    
    # Define specific output directory for this instance
    instance_output_dir="${output_base}/${instance_id}"
    
    echo "[Job $job_id] Starting $instance_id -> $instance_output_dir"
    
    # Run the python script
    # Redirect stdout/stderr to a log file to avoid clutter
    python SE_Perf/perf_run.py \
        --config "$config" \
        --instance "$instance_path" \
        --output_dir "$instance_output_dir" \
        > "${LOG_DIR}/${instance_id}.log" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "[Job $job_id] Finished $instance_id"
    else
        echo "[Job $job_id] Failed $instance_id (Check logs: ${LOG_DIR}/${instance_id}.log)"
    fi
}

# Export function and variables for xargs/parallel use if needed, 
# but here we use a simple background job loop with wait
pids=()
job_count=0

for i in "${!FILES[@]}"; do
    file="${FILES[$i]}"
    
    # Run in background
    process_file "$file" "$CONFIG" "$OUTPUT_BASE" "$i" &
    pids+=($!)
    
    # Control concurrency
    # Count running jobs
    current_jobs=$(jobs -r | wc -l)
    while [ "$current_jobs" -ge "$MAX_JOBS" ]; do
        sleep 1
        current_jobs=$(jobs -r | wc -l)
    done
done

# Wait for all remaining jobs
wait
echo "All jobs completed."
