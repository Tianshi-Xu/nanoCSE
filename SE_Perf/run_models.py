"""
perf_run 数据模型

定义 perf_run 流水线中各模块间传递的结构化数据类型，
替代原先通过裸 dict 传递的非类型安全方式。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrajectoryData:
    """轨迹池条目数据。

    用于 ``traj_pool_manager.summarize_and_add_trajectories()`` 的输入，
    替代原先手动构建的 dict。

    Attributes:
        label: 轨迹标签（如 "sol1", "iter3"）。
        instance_name: 实例名称。
        problem_description: 问题描述文本。
        trajectory_content: 优化轨迹内容（.tra 格式文本）。
        patch_content: 优化后的代码。
        iteration: 迭代编号。
        performance: 最终性能值。
        source_dir: 源输出目录。
        source_entry_labels: 来源轨迹标签列表。
        operator_name: 产生此轨迹的算子名称。
        perf_metrics: 性能指标字典。
    """

    label: str = ""
    instance_name: str = ""
    problem_description: str = ""
    trajectory_content: str = ""
    patch_content: str = ""
    iteration: int = 0
    performance: float | str | None = None
    source_dir: str = ""
    source_entry_labels: list[str] = field(default_factory=list)
    operator_name: str | None = None
    perf_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为 dict，兼容 traj_pool_manager 接口。"""
        return {
            "label": self.label,
            "instance_name": self.instance_name,
            "problem_description": self.problem_description,
            "trajectory_content": self.trajectory_content,
            "patch_content": self.patch_content,
            "iteration": self.iteration,
            "performance": self.performance,
            "source_dir": self.source_dir,
            "source_entry_labels": list(self.source_entry_labels),
            "operator_name": self.operator_name,
            "perf_metrics": dict(self.perf_metrics),
        }


@dataclass
class GlobalMemoryContext:
    """Global Memory 检索上下文。

    替代 ``_retrieve_global_memory()`` 中手动构建的 context dict。

    Attributes:
        language: 编程语言（如 "python3"）。
        optimization_target: 优化目标（如 "runtime"）。
        problem_description: 问题描述。
        additional_requirements: 额外要求文本。
        local_memory: 本地记忆内容。
    """

    language: str = "python3"
    optimization_target: str = "runtime"
    problem_description: str = ""
    additional_requirements: str = ""
    local_memory: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转换为 dict，兼容 GlobalMemoryManager 接口。"""
        return {
            "language": self.language,
            "optimization_target": self.optimization_target,
            "problem_description": self.problem_description,
            "additional_requirements": self.additional_requirements,
            "local_memory": self.local_memory,
        }


@dataclass
class PredictionEntry:
    """单实例预测结果条目。

    替代 ``write_iteration_preds_from_result()`` 中手动构建的预测 dict。

    Attributes:
        code: 优化后的代码。
        passed: 是否通过（性能非 inf）。
        performance: 最终性能值。
        final_metrics: 性能指标字典。
    """

    code: str = ""
    passed: bool = False
    performance: float | str | None = None
    final_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为 dict，用于 JSON 序列化。"""
        return {
            "code": self.code,
            "passed": self.passed,
            "performance": self.performance,
            "final_metrics": dict(self.final_metrics),
        }
