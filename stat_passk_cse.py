import os
import json
import sys

# pass@k 取值
K_LIST = [1, 5, 10, 15, 20, 25, 30]
# 根目录
BASE_DIR = "trajectories_perf/aime_evolve_20260214_161949"

def get_acc_list(preds):
    # 按 iteration 升序排序
    preds_sorted = sorted(preds, key=lambda x: x["iteration"])
    return [item["artifacts"].get("acc", False) for item in preds_sorted]

def main():
    base_dir = sys.argv[1] if len(sys.argv) > 1 else BASE_DIR
    k_list = K_LIST

    passk = {k: 0 for k in k_list}
    total = 0

    # 自动遍历所有二级子目录
    for entry in sorted(os.listdir(base_dir)):
        folder = os.path.join(base_dir, entry)
        if not os.path.isdir(folder):
            continue
        preds_path = os.path.join(folder, "preds.json")
        if not os.path.exists(preds_path):
            continue
        with open(preds_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 用文件夹名作为 key
        if entry in data:
            acc_list = get_acc_list(data[entry])
        elif len(data) == 1:
            # preds.json 中只有一个 key 时直接使用
            acc_list = get_acc_list(next(iter(data.values())))
        else:
            continue
        total += 1
        for k in k_list:
            acc_k = acc_list[:k]
            if any(acc_k):
                passk[k] += 1

    print(f"共统计 {total} 题")
    for k in k_list:
        v = passk[k]
        print(f"pass@{k}: {v}/{total} = {v/total if total else 0:.3f}")

if __name__ == "__main__":
    main()
