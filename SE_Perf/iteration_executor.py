"""
迭代执行模块

负责 PerfAgent 的单次运行、算子执行以及迭代步骤的统一调度。
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

from core.utils.global_memory_manager import GlobalMemoryManager
from core.utils.local_memory_manager import LocalMemoryManager
from core.utils.traj_pool_manager import TrajPoolManager
from operators import create_operator
from operators.base import OperatorContext, OperatorResult
from perf_config import SEPerfRunSEConfig, StepConfig
from results_io import write_iteration_preds_from_result, write_result_json
from run_helpers import build_operator_context, build_perf_agent_config, retrieve_global_memory
from trajectory_handler import process_and_summarize

from perfagent.agent import PerfAgent
from perfagent.config import PerfAgentConfig
from perfagent.protocols import PerfAgentRequest, PerfAgentResult


# ---------------------------------------------------------------------------
# PerfAgent 单次运行
# ---------------------------------------------------------------------------


def run_single_perfagent(
    instance,
    perf_config: PerfAgentConfig,
    operator_result: OperatorResult,
    local_memory_text: str | None,
    global_memory_text: str | None,
    output_dir: Path,
    logger,
) -> PerfAgentResult:
    """构建 PerfAgentRequest 并直接调用 PerfAgent.run_with_request()。"""
    instance_id = getattr(instance, "task_name", None) or getattr(instance, "id", "unknown")

    # 构建配置副本
    config = copy.deepcopy(perf_config)
    config.logging.trajectory_dir = output_dir
    config.logging.log_dir = output_dir

    request = PerfAgentRequest(
        instance=instance,
        config=config,
        initial_code=operator_result.initial_code,
        additional_requirements=operator_result.additional_requirements,
        local_memory=local_memory_text,
        global_memory=global_memory_text,
        output_dir=output_dir,
    )

    logger.info(
        f"构建 PerfAgentRequest: instance={instance_id}, "
        f"has_additional_requirements={bool(operator_result.additional_requirements)}, "
        f"has_initial_code={bool(operator_result.initial_code)}, "
        f"has_local_memory={bool(local_memory_text)}, "
        f"has_global_memory={bool(global_memory_text)}"
    )

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        agent = PerfAgent(config)
        result = agent.run_with_request(request)
        logger.info(
            f"PerfAgent 执行完成: instance={instance_id}, "
            f"success={result.success}, "
            f"performance={result.final_performance}"
        )
        return result
    except Exception as e:
        logger.error(f"PerfAgent 执行异常: {e}", exc_info=True)
        return PerfAgentResult.from_error(instance_id=instance_id, error=str(e))


# ---------------------------------------------------------------------------
# 算子执行
# ---------------------------------------------------------------------------


def run_operator(
    step: StepConfig,
    instance_name: str,
    instance_entry: dict[str, Any],
    op_context: OperatorContext,
    traj_pool_manager: TrajPoolManager,
    logger,
) -> list[OperatorResult]:
    """执行算子并返回 OperatorResult 列表。

    - Plan 算子返回多个 OperatorResult。
    - Filter 算子直接修改轨迹池后返回空列表（不触发 PerfAgent）。
    - 普通算子返回单元素列表。
    - 无算子时返回默认空 OperatorResult 的单元素列表。
    """
    if not step.operator:
        return [OperatorResult()]

    logger.info(f"执行算子: {step.operator}")
    op_instance = create_operator(step.operator, op_context)
    if not op_instance:
        logger.error(f"无法创建算子实例: {step.operator}")
        return [OperatorResult()]

    try:
        if step.is_filter:
            op_instance.run_for_instance(
                step, instance_name, instance_entry,
                traj_pool_manager=traj_pool_manager,
            )
            logger.info("Filter 算子执行完毕，跳过后续 PerfAgent 运行")
            return []

        result = op_instance.run_for_instance(step, instance_name, instance_entry)
        if isinstance(result, list):
            return result  # Plan 算子返回 list[OperatorResult]
        return [result]
    except Exception as e:
        logger.error(f"算子 '{step.operator}' 执行失败: {e}")
        return [OperatorResult()]


def resolve_labels(step: StepConfig, num_results: int) -> list[str]:
    """为算子结果列表生成对应的轨迹标签。"""
    if step.is_plan:
        labels = list(step.trajectory_labels or [])
        if not labels:
            labels = [f"sol{i}" for i in range(1, num_results + 1)]
        # 补齐不足的标签
        while len(labels) < num_results:
            labels.append(f"sol{len(labels) + 1}")
        return labels[:num_results]
    else:
        # 普通算子：使用 trajectory_label 或默认标签
        label = step.trajectory_label
        return [label] * num_results if label else [None] * num_results  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# 单次执行（PerfAgent + 后处理）
# ---------------------------------------------------------------------------


def execute_single_run(
    op_result: OperatorResult,
    label: str | None,
    instance,
    instance_name: str,
    se_cfg: SEPerfRunSEConfig,
    step: StepConfig,
    traj_pool_manager: TrajPoolManager,
    local_memory_text: str | None,
    global_memory: GlobalMemoryManager | None,
    default_lang: str,
    default_target: str,
    output_dir: str,
    iteration_idx: int,
    mode: str,
    logger,
) -> None:
    """执行单次 PerfAgent 运行并后处理。"""
    iter_dir = Path(output_dir) / f"iteration_{iteration_idx}"
    instance_output_dir = iter_dir / instance_name

    # 检索 Global Memory
    global_memory_text = retrieve_global_memory(
        global_memory=global_memory,
        instance_name=instance_name,
        problem_description=instance.description_md or "",
        additional_requirements=op_result.additional_requirements,
        local_memory_text=local_memory_text,
        default_lang=default_lang,
        default_target=default_target,
    )

    # 构建 PerfAgent 配置
    perf_config = build_perf_agent_config(
        base_config_path=step.perf_base_config or se_cfg.base_config,
        se_model_config=se_cfg.model.to_dict(),
        max_iterations=se_cfg.max_iterations,
        output_dir=instance_output_dir,
    )

    label_info = f" ({step.operator}: {label})" if step.operator and label else ""
    print(f"\n=== 迭代 {iteration_idx}{label_info} ===")
    os.environ["SE_ITERATION_INDEX"] = str(iteration_idx)

    if mode == "demo":
        logger.info("演示模式：跳过实际执行")
        return

    # 执行 PerfAgent
    result = run_single_perfagent(
        instance=instance,
        perf_config=perf_config,
        operator_result=op_result,
        local_memory_text=local_memory_text,
        global_memory_text=global_memory_text,
        output_dir=instance_output_dir,
        logger=logger,
    )

    # 写入文件（用于持久化和调试）
    write_result_json(result, instance_output_dir, logger)
    write_iteration_preds_from_result(result, iter_dir, logger)

    # 后处理：更新轨迹池
    if result.success or result.error is None:
        source_labels_map = (
            {instance_name: op_result.source_labels} if op_result.source_labels else None
        )
        process_and_summarize(
            iter_dir,
            iteration_idx,
            step,
            se_cfg,
            traj_pool_manager,
            logger,
            label_prefix=label,
            source_labels_map=source_labels_map,
            operator_name=step.operator,
            result=result,
            instance_name=instance_name,
            problem_description=instance.description_md or "",
        )

    if op_result.source_labels:
        logger.info(
            f"算子 {step.operator} 执行完成: "
            f"has_additional_requirements={bool(op_result.additional_requirements)}, "
            f"has_initial_code={bool(op_result.initial_code)}, "
            f"source_labels={op_result.source_labels}"
        )


# ---------------------------------------------------------------------------
# 迭代步骤入口
# ---------------------------------------------------------------------------


def execute_iteration(
    step: StepConfig,
    instance,
    instance_name: str,
    se_cfg: SEPerfRunSEConfig,
    traj_pool_manager: TrajPoolManager,
    local_memory: LocalMemoryManager | None,
    global_memory: GlobalMemoryManager | None,
    default_lang: str,
    default_target: str,
    output_dir: str,
    iteration_idx: int,
    mode: str,
    logger,
) -> int:
    """执行单个迭代步骤，返回下一个迭代索引。

    统一了 Plan / Filter / Regular 三个分支的执行逻辑。
    """
    # 刷新 local_memory_text
    local_memory_text = None
    try:
        if local_memory is not None:
            _mem_latest = local_memory.load()
            local_memory_text = local_memory.render_as_markdown(_mem_latest)
    except Exception:
        pass

    # 获取当前实例在轨迹池中的条目
    instance_entry = traj_pool_manager.get_instance(instance_name) or {}

    # 构建算子上下文
    op_context = build_operator_context(se_cfg, step)

    # 执行算子
    op_results = run_operator(step, instance_name, instance_entry, op_context, traj_pool_manager, logger)

    # Filter 算子不触发 PerfAgent，直接返回
    if step.is_filter or not op_results:
        return iteration_idx

    # 为每个结果生成对应标签
    labels = resolve_labels(step, len(op_results))

    # 对每个 OperatorResult 执行 PerfAgent + 后处理
    for op_result, label in zip(op_results, labels):
        execute_single_run(
            op_result=op_result,
            label=label,
            instance=instance,
            instance_name=instance_name,
            se_cfg=se_cfg,
            step=step,
            traj_pool_manager=traj_pool_manager,
            local_memory_text=local_memory_text,
            global_memory=global_memory,
            default_lang=default_lang,
            default_target=default_target,
            output_dir=output_dir,
            iteration_idx=iteration_idx,
            mode=mode,
            logger=logger,
        )
        iteration_idx += 1

    return iteration_idx
