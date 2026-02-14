#!/bin/bash

# Configuration
INPUT_DATASET="instances"              # Directory containing original JSON problems
SCORED_DIR="instances/aime2025_scored"        # Directory to save scored/evaluated problems
FILTERED_DIR="instances/aime2025_filtered"        # Directory to save filtered (hard) problems
ROLLOUTS=16                            # Number of solutions to generate per problem
THREADS=8                              # Number of concurrent threads
MAX_PASS_RATE=0.2
MIN_PASS_RATE=0.0

echo "Step 2: Filtering hard instances (pass_rate <= $MAX_PASS_RATE)..."
python utils/filter_scored_instances.py \
    --input_dir "$SCORED_DIR" \
    --output_dir "$FILTERED_DIR" \
    --min_pass_rate "$MIN_PASS_RATE" \
    --max_pass_rate "$MAX_PASS_RATE"

echo "Done! Filtered dataset is in $FILTERED_DIR"
