#!/usr/bin/env python3
"""
PerfAgent 单实例集成执行脚本

功能：
    在 SE 框架中驱动 PerfAgent 对 **单个实例** 进行多次迭代的性能优化。
    所有 SE_Perf 与 PerfAgent 之间的信息传递通过 PerfAgentRequest / PerfAgentResult 完成，
    不再使用文件系统作为中间通道。
"""

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

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

from perfagent.protocols import PerfAgentRequest, PerfAgentResult
from perfagent.run import load_instance_data

# 从拆分模块导入功能函数
from iteration_executor import execute_iteration
from results_io import log_token_usage, print_final_summary
from run_helpers import retrieve_global_memory, build_perf_agent_config

# ----------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main():
    """主函数：单实例模式的 PerfAgent 多迭代执行入口。"""
    parser = argparse.ArgumentParser(description="SE 框架 PerfAgent 单实例多迭代执行脚本")
    parser.add_argument("--config", default="configs/Plan-Weighted-Local-Global-30.yaml", help="SE 配置文件路径")
    parser.add_argument("--instance", required=True, help="单个实例 JSON 文件路径")
    parser.add_argument("--mode", choices=["demo", "execute"], default="execute", help="运行模式")
    args = parser.parse_args()

    print("=== CSE 单实例执行 ===")

    try:
        # 1. 加载配置
        with open(args.config, encoding="utf-8") as f:
            se_raw = yaml.safe_load(f) or {}
        se_cfg = SEPerfRunSEConfig.from_dict(se_raw)

        # 2. 加载单实例
        instance_path = Path(args.instance)
        if not instance_path.exists():
            print(f"实例文件不存在: {instance_path}")
            sys.exit(1)

        instance = load_instance_data(instance_path)
        instance_name = instance.task_name or instance.id
        print(f"  实例: {instance_name}")

        # 3. 准备输出环境
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = se_cfg.output_dir.replace("{timestamp}", timestamp)

        # 如果 final.json 存在，认为任务已完成
        if (Path(output_dir) / "final.json").exists():
            log_file = setup_se_logging(output_dir)
            logger = get_se_logger("perf_run", emoji="⚡")
            print("检测到任务已完成，跳过执行")
            logger.info("检测到任务已完成，直接结束")
            log_token_usage(output_dir, logger)
            return

        # 未完成：清空输出目录并初始化日志
        try:
            if Path(output_dir).exists():
                shutil.rmtree(output_dir)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"清空输出目录失败: {e}")

        log_file = setup_se_logging(output_dir)
        logger = get_se_logger("perf_run", emoji="⚡")

        logger.info(f"启动执行: config={args.config}, instance={args.instance}, 模式: {args.mode}")
        logger.info(f"实例名称: {instance_name}")
        logger.info(f"输出目录: {output_dir}")

        # Token统计与LLM I/O日志文件路径
        os.environ["SE_TOKEN_LOG_PATH"] = str(Path(output_dir) / "token_usage.jsonl")
        os.environ["SE_LLM_IO_LOG_PATH"] = str(Path(output_dir) / "llm_io.jsonl")

        # 4. 初始化核心组件

        # LLM Client
        llm_client = None
        try:
            from core.utils.llm_client import LLMClient

            llm_client = LLMClient(se_cfg.model.to_dict())
        except Exception as e:
            logger.warning(f"LLM客户端初始化失败: {e}")

        # Local Memory Manager
        local_memory = None
        memory_config = se_cfg.local_memory
        if isinstance(memory_config, LocalMemoryConfig) and memory_config.enabled:
            try:
                memory_path = Path(output_dir) / "memory.json"
                local_memory = LocalMemoryManager(
                    memory_path,
                    llm_client=llm_client,
                    format_mode=memory_config.format_mode,
                )
                local_memory.initialize()
                logger.info("LocalMemoryManager 已启用")
            except Exception as e:
                logger.warning(f"LocalMemoryManager 初始化失败: {e}")

        # Trajectory Pool Manager
        traj_pool_path = str(Path(output_dir) / "traj.pool")
        traj_pool_manager = TrajPoolManager(
            traj_pool_path,
            llm_client,
            num_workers=se_cfg.num_workers,
            memory_manager=local_memory,
            prompt_config=se_cfg.prompt_config,
        )
        traj_pool_manager.initialize_pool()

        # Global Memory Manager
        global_memory = None
        global_memory_config = se_cfg.global_memory_bank
        if isinstance(global_memory_config, GlobalMemoryConfig) and global_memory_config.enabled:
            try:
                global_memory = GlobalMemoryManager(llm_client=llm_client, bank_config=global_memory_config)
                logger.info("GlobalMemoryManager 已启用")
            except Exception as e:
                logger.warning(f"GlobalMemoryManager 初始化失败: {e}")

        # 读取 base_config 中的默认语言和优化目标（用于 global memory 检索）
        default_lang = "python3"
        default_target = "runtime"
        base_config_path = se_cfg.base_config
        if base_config_path:
            try:
                with open(base_config_path, encoding="utf-8") as f:
                    bc = yaml.safe_load(f) or {}
                    default_lang = bc.get("language_cfg", {}).get("language", default_lang)
                    default_target = bc.get("optimization", {}).get("target", default_target)
            except Exception:
                pass

        # 5. 执行迭代策略
        iterations = se_cfg.strategy.iterations
        logger.info(f"计划执行 {len(iterations)} 个迭代步骤")

        next_iteration_idx = 1

        for step in iterations:
            next_iteration_idx = execute_iteration(
                step=step,
                instance=instance,
                instance_name=instance_name,
                se_cfg=se_cfg,
                traj_pool_manager=traj_pool_manager,
                local_memory=local_memory,
                global_memory=global_memory,
                default_lang=default_lang,
                default_target=default_target,
                output_dir=output_dir,
                iteration_idx=next_iteration_idx,
                mode=args.mode,
                logger=logger,
            )

        # Update global memory
        if global_memory:
            global_memory.update_from_pool(traj_pool_manager)

        # 6. 最终汇总
        print_final_summary(timestamp, log_file, output_dir, traj_pool_manager, logger)

    except Exception as e:
        if "logger" in locals():
            logger.error(f"程序运行异常: {e}", exc_info=True)
        print(f"程序运行异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
