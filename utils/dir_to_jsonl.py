import json
from pathlib import Path
import sys

def convert_dir_to_jsonl(input_dir, output_file):
    path = Path(input_dir)
    files = list(path.glob("**/*.json"))
    print(f"Found {len(files)} JSON files in {input_dir}")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for p in files:
            if "manifest" in p.name or "report" in p.name or "sample" in p.name:
                continue
            try:
                content = json.loads(p.read_text(encoding="utf-8"))
                # Ensure 'id' exists logic if needed, but assuming format is OK
                f.write(json.dumps(content, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Skipping {p}: {e}")
    print(f"Created {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert.py <input_dir> <output_jsonl>")
        sys.exit(1)
    convert_dir_to_jsonl(sys.argv[1], sys.argv[2])
