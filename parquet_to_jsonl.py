#!/usr/bin/env python3
"""
Convert Parquet files to JSONL format.
Each row in the Parquet file becomes one JSON object per line in the output file.
"""

import argparse
import json
from pathlib import Path
import pandas as pd


def parquet_to_jsonl(parquet_path: str, jsonl_path: str = None):
    """
    Convert a Parquet file to JSONL format.
    
    Args:
        parquet_path: Path to the input Parquet file
        jsonl_path: Path to the output JSONL file (optional, defaults to same name with .jsonl extension)
    """
    parquet_path = Path(parquet_path)
    
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    
    # Generate output path if not provided
    if jsonl_path is None:
        jsonl_path = parquet_path.with_suffix('.jsonl')
    else:
        jsonl_path = Path(jsonl_path)
    
    # Create output directory if it doesn't exist
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading Parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    print(f"Found {len(df)} rows with {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    print(f"Writing to JSONL file: {jsonl_path}")
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            json_obj = row.to_dict()
            # Convert to JSON and write one line per record
            f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
    
    print(f"Successfully converted {len(df)} rows to {jsonl_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert Parquet files to JSONL format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert with automatic output name
  python parquet_to_jsonl.py instances/Open-AgentRL-Eval/aime2024/aime_2024_problems.parquet
  
  # Specify output file
  python parquet_to_jsonl.py input.parquet -o output.jsonl
        """
    )
    
    parser.add_argument(
        'parquet_file',
        type=str,
        help='Path to the input Parquet file'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Path to the output JSONL file (default: same name as input with .jsonl extension)'
    )
    
    args = parser.parse_args()
    
    try:
        parquet_to_jsonl(args.parquet_file, args.output)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

