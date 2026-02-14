import argparse
import json
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Filter scored instances by pass rate from JSONL")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file (output from filter_aime.py)")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file for filtered instances")
    parser.add_argument("--min_pass_rate", type=float, default=0.0, help="Minimum pass rate (inclusive)")
    parser.add_argument("--max_pass_rate", type=float, default=1.0, help="Maximum pass rate (inclusive)")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Filtering {input_path} -> {output_path}")
    print(f"Criteria: {args.min_pass_rate} <= pass_rate <= {args.max_pass_rate}")
    
    count = 0
    total = 0
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip(): continue
            total += 1
            try:
                data = json.loads(line)
                # Extract pass_rate from metadata or root
                pass_rate = data.get("metadata", {}).get("pass_rate")
                if pass_rate is None:
                    pass_rate = data.get("pass_rate") # Fallback
                
                if pass_rate is not None:
                    if args.min_pass_rate <= float(pass_rate) <= args.max_pass_rate:
                        fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                        count += 1
            except Exception as e:
                print(f"Error processing line: {e}")
            
    print(f"Filtered {count}/{total} instances to {output_path}")

if __name__ == "__main__":
    main()
