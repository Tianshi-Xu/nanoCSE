#!/usr/bin/env python3
"""
Filter AIME instances based on pass rate.
Run N rollouts, if pass rate is 100%, filter out.
"""

import argparse
import json
import logging
import re
import sys
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from perfagent.llm_client import LLMClient
from perfagent.tasks.math_dapo import compute_score, last_boxed_only_string, remove_boxed
from perfagent.tasks.aime import AIMEInstance

# Logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default Configuration
DEFAULT_CONFIG = {
    "model": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "api_base": "http://127.0.0.1:37993/v1",
        "api_key": "",
        # "name": "qwen/qwen3-4b:free",
        # "api_base": "https://openrouter.ai/api/v1",
        # "api_key": "",
        "max_output_tokens": 16384,
        "temperature": 0.7,
        "top_k": 20,
        "top_p": 0.95
    },
    "system_template": """You are an expert competition mathematician solving an AIME-style problem. Analyze and solve the following math problem step by step. \n\n
{problem_statement}

## Instructions
1. Solve the problem carefully with clear reasoning.
2. The final answer should be a single integer (AIME format 0-999).
3. Output the final answer in the following text format, the answer format must be: \\boxed{{'The final answer goes here.'}}
"""
}

def extract_solution(llm_response: str) -> str:
    # Copied/Adapted from AIMERunner
    answer_matches = re.findall(r"(?i)Answer\s*:\s*([^\n]+)", llm_response)
    if answer_matches:
        answer = answer_matches[-1].strip()
        return f"Answer: {answer}"

    boxed = last_boxed_only_string(llm_response)
    if boxed:
        try:
            unboxed = remove_boxed(boxed)
            return f"Answer: {unboxed}"
        except Exception:
            return f"Answer: {boxed}"
            
    return ""

def determine_failure_reason(llm_response: str) -> str:
    """Analyze why extraction failed."""
    if not llm_response:
        return "empty_response"
        
    # Check for truncation indicators (heuristic)
    if len(llm_response) > 0 and llm_response[-1] not in ['.', '!', '?', '>', '}']:
        # This is a weak heuristic, but often truncated text ends abruptly
        # Better check might be checking if the finish_reason of the API response was 'length'
        # But we only have the text here.
        pass
        
    return "no_format_match"

def process_instance_data(instance: AIMEInstance, client: LLMClient, args) -> dict:
    if not instance.answer:
        return {"id": instance.id, "error": "no_answer"}

    prompt = DEFAULT_CONFIG["system_template"].format(problem_statement=instance.problem)
    messages = [{"role": "system", "content": prompt}]
    
    pass_count = 0
    truncation_count = 0
    results = []
    
    for _ in range(args.rollouts):
        try:
            resp_data = client.call_llm(
                messages, 
                temperature=DEFAULT_CONFIG["model"].get("temperature", 0.7), 
                max_tokens=DEFAULT_CONFIG["model"].get("max_output_tokens"),
                return_full_response=True,
                extra_params={
                    "top_k": DEFAULT_CONFIG["model"].get("top_k"),
                    "top_p": DEFAULT_CONFIG["model"].get("top_p")
                },
                stream=True
            )
            resp = resp_data["content"]
            finish_reason = resp_data.get("finish_reason", "unknown")
            
            if finish_reason == "length":
                truncation_count += 1

            extracted = extract_solution(resp)
            if not extracted:
                 # Extraction failed
                 failure_reason = determine_failure_reason(resp)
                 if finish_reason == "length":
                     failure_reason = "truncated_length"
                 
                 results.append({
                     "correct": False, 
                     "extracted": "", 
                     "failure_reason": failure_reason,
                     "finish_reason": finish_reason,
                     "response_preview": resp[-200:] if resp else "" 
                 })
                 continue

            score_dict = compute_score(extracted, instance.answer)
            is_correct = bool(score_dict.get("acc", False))
            if is_correct:
                pass_count += 1
            results.append({"correct": is_correct, "extracted": extracted, "finish_reason": finish_reason})
        except Exception as e:
            logger.error(f"Error in rollout for {instance.id}: {e}")

    pass_rate = pass_count / args.rollouts if args.rollouts > 0 else 0
    pass_str = f"{pass_count}/{args.rollouts}"
    truncation_rate = truncation_count / args.rollouts if args.rollouts > 0 else 0
    
    # Update instance metadata with stats
    data = {"id": instance.id, "problem": instance.problem, "answer": instance.answer, "metadata": instance.metadata}
    data["metadata"]["pass_rate"] = pass_rate
    data["metadata"]["pass_stats"] = pass_str

    details = {
        "id": instance.id, 
        "pass_rate": pass_rate, 
        "pass_stats": pass_str, 
        "truncation_rate": truncation_rate,
        "results": results,
    }
    return data, details

def calculate_pass_at_k(n, c, k):
    """
    Calculate pass@k using the unbiased estimator.
    n: total number of samples
    c: number of correct samples
    k: k in pass@k
    Score = 1 - combin(n-c, k) / combin(n, k)
    """
    import math
    if n - c < k:
        return 1.0
    
    # Calculate combin(n, k)
    # combin(n, k) = n! / (k! * (n-k)!)
    # We want 1 - ( (n-c)! / (k! * (n-c-k)!) ) / ( n! / (k! * (n-k)!) )
    #           = 1 - [ (n-c)! * (n-k)! ] / [ (n-c-k)! * n! ]
    #           = 1 - product((n-c-i)/(n-i) for i in range(k))
    
    prob_all_wrong = 1.0
    for i in range(k):
        prob_all_wrong *= (n - c - i) / (n - i)
        
    return 1.0 - prob_all_wrong

def main():
    parser = argparse.ArgumentParser(description="Evaluate AIME instances and add pass_rate metadata")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file containing instances")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file for processed instances")
    parser.add_argument("--rollouts", type=int, default=4, help="Number of rollouts per instance")
    parser.add_argument("--threads", type=int, default=8, help="Concurrent threads")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = LLMClient(model_config=DEFAULT_CONFIG["model"])
    
    print("-" * 50)
    print(f"Model: {DEFAULT_CONFIG['model']['name']}")
    print(f"Input File: {args.input_file}")
    print(f"Output File: {args.output_file}")
    print(f"Rollouts per instance: {args.rollouts}")
    print("-" * 50)
    
    instances = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    instances.append(AIMEInstance.from_dict(json.loads(line), input_path))
                except json.JSONDecodeError:
                    continue
    
    print(f"Found {len(instances)} instances. Starting evaluation with {args.rollouts} rollouts...")

    processed_count = 0
    report_data = []
    results_list_all = [] # To store updated instances

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(process_instance_data, inst, client, args): inst for inst in instances}
        
        for future in tqdm(as_completed(futures), total=len(instances)):
            try:
                data, details = future.result()
                if "error" in details or "error" in data:
                    continue
                    
                report_data.append(details)
                results_list_all.append(data)
                
                processed_count += 1
            except Exception as e:
                pass

    # Save all updated instances to output_file
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results_list_all:
             f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Write report (beside output file)
    report_path = output_path.with_name(output_path.stem + "_report.json")
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)

    # Analyze failures & Stats
    failure_counts = {}
    total_failures = 0
    total_truncations = 0
    total_rollouts = 0
    
    # Validation stats
    total_instances = len(report_data)
    total_correct_once = 0  # At least one correct
    total_pass_1_sum = 0.0  # Sum of pass rates (average is pass@1)
    
    # 统计 pass@1 到 pass@rollouts，每隔1个都统计
    k_values = list(range(1, args.rollouts + 1))
    pass_at_k_sums = {k: 0.0 for k in k_values}

    for item in report_data:
        results_list = item.get("results", [])
        n_samples = len(results_list)
        n_correct = sum(1 for r in results_list if r.get("correct", False))

        total_rollouts += n_samples

        # pass@1 is effectively the平均准确率
        if n_samples > 0:
            total_pass_1_sum += (n_correct / n_samples)

        # pass@k, k从1到rollouts
        for k in k_values:
            score = calculate_pass_at_k(n_samples, n_correct, k)
            pass_at_k_sums[k] += score

        if n_correct > 0:
            total_correct_once += 1

        # Count truncations across all outcomes
        for res in results_list:
            if res.get("finish_reason") == "length":
                total_truncations += 1

        for res in results_list:
            if not res.get("correct", False) and not res.get("extracted"):
                reason = res.get("failure_reason", "unknown")
                failure_counts[reason] = failure_counts.get(reason, 0) + 1
                total_failures += 1
    
    print("\n" + "="*40)
    print(f"EVALUATION REPORT (n={args.rollouts})")
    print("="*40)
    
    summary_stats = {
        "model_name": DEFAULT_CONFIG["model"]["name"],
        "dataset": str(args.input_file),
        "rollouts": args.rollouts,
        "total_instances": total_instances
    }

    if total_instances > 0:
        print(f"Total Instances: {total_instances}")

        # Metric 1: Pass@1 (Mean of pass rates)
        pass_at_1 = total_pass_1_sum / total_instances
        print(f"Pass@1 (Mean Accuracy): {pass_at_1:.4f}")
        summary_stats["pass@1"] = pass_at_1

        # Metric 2: Pass@k (Unbiased estimator), k=1到rollouts
        for k in k_values:
            pk = pass_at_k_sums[k] / total_instances
            print(f"Pass@{k}: {pk:.4f}")
            summary_stats[f"pass@{k}"] = pk

        print(f"Solved (at least 1 correct): {total_correct_once}/{total_instances} ({total_correct_once/total_instances*100:.2f}%)")
        summary_stats["solved_any"] = total_correct_once
        summary_stats["solved_percentage"] = total_correct_once/total_instances

    # Save statistics to a summary file
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_stats, f, indent=2)

    if total_failures > 0 or total_truncations > 0:
        print("\n--- Failure & Truncation Analysis ---")
        if total_rollouts > 0:
            print(f"  Total Rollouts: {total_rollouts}")
            print(f"  Truncation Events: {total_truncations} ({total_truncations/total_rollouts*100:.2f}%)")
        
        print("  Extraction Failures by Reason:")
        for reason, count in failure_counts.items():
            print(f"    {reason}: {count}")

if __name__ == "__main__":
    main()
