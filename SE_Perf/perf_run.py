#!/usr/bin/env python3
"""
PerfAgent 单实例集成执行脚本

功能：
    在 SE 框架中驱动 PerfAgent 对单个实例进行多次迭代的性能优化。
    所有 SE_Perf 与 PerfAgent 之间的信息传递通过 AgentRequest / AgentResult 完成，
    不再使用文件系统作为中间通道。
"""

import argparse
import os

import sys
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import yaml

# 添加 SE 根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

# 导入 SE 核心模块
from core.global_memory.utils.config import GlobalMemoryConfig
from core.utils.global_memory_manager import GlobalMemoryManager
from core.utils.local_memory_manager import LocalMemoryManager
from core.utils.se_logger import get_se_logger, setup_se_logging
from core.utils.traj_pool_manager import TrajPoolManager
from perf_config import LocalMemoryConfig, SEPerfRunSEConfig

from perfagent.task_registry import create_task_runner

# 从拆分模块导入功能函数
from iteration_executor import execute_iteration
from results_io import log_token_usage, print_final_summary
from run_helpers import retrieve_global_memory, build_perf_agent_config, load_metric_higher_is_better

import json

# ----------------------------------------------------------------------------
# 单实例执行逻辑
# ---------------------------------------------------------------------------

def run_single_instance(instance_data, se_cfg, output_base, task_runner=None, mode="execute"):
    """
    处理单个实例的优化过程

    Args:
        instance_data: 实例原始字典数据
        se_cfg: SE 配置对象
        output_base: 输出根目录
        task_runner: 任务 runner 实例（由调用方创建，避免重复实例化）
        mode: 运行模式 ("execute" / "demo")
    """
    # 通过 task_runner 提取元数据（无需文件 I/O）
    task_type = se_cfg.task_type or "effibench"
    if task_runner is None:
        task_runner = create_task_runner(task_type)

    metadata = task_runner.load_metadata_from_dict(instance_data)
    instance_id = metadata.instance_id
    problem_description = metadata.problem_description or ""

    instance_safe_id = str(instance_id).replace("/", "_")
    output_dir = str(Path(output_base) / instance_safe_id)

    # 如果 final.json 存在，认为任务已完成
    if (Path(output_dir) / "final.json").exists():
        logger = get_se_logger(f"perf_run_{instance_safe_id}", emoji="⚡")
        logger.info(f"Instance {instance_id} exists. Skipping.")
        return

    # setup logging
    log_file = setup_se_logging(output_dir)
    logger = get_se_logger(f"perf_run_{instance_safe_id}", emoji="⚡")

    logger.info(f"Start Instance: {instance_id}")

    # 写入临时文件供下游 PerfAgent（通过 task_data_path）使用
    temp_instance_dir = Path(output_dir) / "_temp_input"
    temp_instance_dir.mkdir(parents=True, exist_ok=True)
    temp_instance_path = temp_instance_dir / f"{instance_safe_id}.json"

    with open(temp_instance_path, "w", encoding="utf-8") as f:
        json.dump(instance_data, f, ensure_ascii=False, indent=2)

    try:
        # Token统计与LLM I/O日志文件路径
        os.environ["SE_TOKEN_LOG_PATH"] = str(Path(output_dir) / "token_usage.jsonl")
        os.environ["SE_LLM_IO_LOG_PATH"] = str(Path(output_dir) / "llm_io.jsonl")

        # LLM Client
        from core.utils.llm_client import LLMClient
        llm_client = LLMClient(se_cfg.model.to_dict())

        # Local Memory
        local_memory = None
        memory_config = se_cfg.local_memory
        if isinstance(memory_config, LocalMemoryConfig) and memory_config.enabled:
            memory_path = Path(output_dir) / "memory.json"
            local_memory = LocalMemoryManager(
                memory_path,
                llm_client=llm_client,
                format_mode=memory_config.format_mode,
            )
            local_memory.initialize()

        # Trajectory Pool
        metric_higher_is_better = load_metric_higher_is_better(se_cfg.base_config)
        traj_pool_path = str(Path(output_dir) / "traj.pool")
        traj_pool_manager = TrajPoolManager(
            traj_pool_path,
            instance_name=str(instance_id),
            llm_client=llm_client,
            memory_manager=local_memory,
            prompt_config=se_cfg.prompt_config.to_dict(),
            metric_higher_is_better=metric_higher_is_better,
        )
        traj_pool_manager.initialize_pool()

        # Global Memory
        global_memory = None
        global_memory_config = se_cfg.global_memory_bank
        if isinstance(global_memory_config, GlobalMemoryConfig) and global_memory_config.enabled:
            global_memory = GlobalMemoryManager(llm_client=llm_client, bank_config=global_memory_config)

        # Execute Iterations
        iterations = se_cfg.strategy.iterations
        logger.info(f"Executing {len(iterations)} iterations for {instance_id}")
        
        next_iteration_idx = 1
        for step in iterations:
            next_iteration_idx = execute_iteration(
                step=step,
                task_data_path=temp_instance_path,
                instance_id=str(instance_id),
                problem_description=problem_description,
                se_cfg=se_cfg,
                traj_pool_manager=traj_pool_manager,
                local_memory=local_memory,
                global_memory=global_memory,
                output_dir=output_dir,
                iteration_idx=next_iteration_idx,
                mode=mode,
                logger=logger,
                task_runner=task_runner,
            )

        if global_memory:
            global_memory.update_from_pool(traj_pool_manager)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print_final_summary(timestamp, log_file, output_dir, traj_pool_manager, logger, metric_higher_is_better)

    except Exception as e:
        logger.error(f"Instance {instance_id} failed: {e}", exc_info=True)
    finally:
        # Cleanup temp file
        if temp_instance_path.exists():
            try:
                os.remove(temp_instance_path)
                temp_instance_dir.rmdir()
            except:
                pass

def run_task_wrapper(args):
    """用于多进程执行的包装函数"""
    instance, se_cfg, output_dir, mode = args
    try:
        # 多进程环境下每个 worker 创建独立的 task_runner
        task_type = se_cfg.task_type or "effibench"
        task_runner = create_task_runner(task_type)
        run_single_instance(instance, se_cfg, output_dir, task_runner=task_runner, mode=mode)
    except Exception as e:
        print(f"Worker process error for instance {instance.get('id', 'unknown')}: {e}")

# ----------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main():
    """主函数：PerfAgent 多迭代执行入口（支持单文件或JSONL批量）。"""
    parser = argparse.ArgumentParser(description="SE 框架 PerfAgent 执行脚本")
    parser.add_argument("--config", default="configs/Plan-Weighted-Local-Global-30.yaml", help="SE 配置文件路径")
    parser.add_argument("--input_file", required=True, help="输入文件路径 (JSON or JSONL)")
    parser.add_argument("--mode", choices=["demo", "execute"], default="execute", help="运行模式")
    parser.add_argument("--workers", type=int, default=1, help="并行工作进程数 (默认为1，即串行)")
    args = parser.parse_args()

    print("=== CSE 批量执行启动 ===")

    # 1. 加载配置
    with open(args.config, encoding="utf-8") as f:
        se_raw = yaml.safe_load(f) or {}
    se_cfg = SEPerfRunSEConfig.from_dict(se_raw)

    # 2. 创建任务 runner 并通过其接口读取输入（支持 JSON / JSONL）
    task_type = se_cfg.task_type or "effibench"
    task_runner = create_task_runner(task_type)

    input_path = Path(args.input_file)
    instances = task_runner.load_all_from_file(input_path)
    print(f"加载了 {len(instances)} 个实例 (task_type={task_type})")

    # 3. 循环执行
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = se_cfg.output_dir.replace("{timestamp}", timestamp)

    if args.workers > 1:
        print(f"启用并行执行，进程数: {args.workers}")
        # 准备任务参数（多进程下每个 worker 会创建独立的 task_runner）
        task_args = [(instance, se_cfg, output_dir, args.mode) for instance in instances]
        
        # 使用 ProcessPoolExecutor 进行多进程执行，确保 os.environ 隔离
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = executor.map(run_task_wrapper, task_args)
            
            count = 0
            for _ in results:
                count += 1
                if count % max(1, len(instances) // 10) == 0:
                    print(f"进度: {count}/{len(instances)} 完成")
    else:
        # 串行执行（复用同一个 task_runner 实例）
        for i, instance in enumerate(instances):
            meta = task_runner.load_metadata_from_dict(instance)
            print(f"[{i+1}/{len(instances)}] Processing {meta.instance_id}...")
            run_single_instance(instance, se_cfg, output_dir, task_runner=task_runner, mode=args.mode)
            
    print("所有任务执行完毕")

if __name__ == "__main__":
    main()
