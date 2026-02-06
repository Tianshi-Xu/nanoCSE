"""
PerfAgent 协议定义

定义 SE_Perf 与 PerfAgent 之间的数据传输接口，
使两个模块之间的通信标准化、清晰化。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent import EffiBenchXInstance
    from .config import PerfAgentConfig


@dataclass
class PerfAgentRequest:
    """PerfAgent 的输入请求

    这是 SE_Perf 传递给 PerfAgent 的标准化请求对象。
    包含实例信息、配置以及可选的覆盖参数。

    Attributes:
        instance: 问题实例数据
        config: PerfAgent 配置
        initial_code: 可选的初始代码覆盖
        additional_requirements: 可选的额外 prompt 要求（来自 Operator）
        local_memory: 可选的本地记忆内容
        global_memory: 可选的全局记忆内容
        output_dir: 可选的输出目录（用于保存轨迹等文件）
    """

    instance: EffiBenchXInstance
    config: PerfAgentConfig
    # 可选覆盖参数
    initial_code: str | None = None
    additional_requirements: str | None = None
    local_memory: str | None = None
    global_memory: str | None = None
    output_dir: Path | None = None

    def apply_overrides(self) -> None:
        """将请求中的覆盖参数应用到配置中"""
        if self.additional_requirements:
            self.config.prompts.additional_requirements = self.additional_requirements
        if self.initial_code:
            self.config.overrides.initial_code_text = self.initial_code
        if self.local_memory:
            self.config.prompts.local_memory = self.local_memory
        if self.global_memory:
            self.config.prompts.global_memory = self.global_memory

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "instance": self.instance.to_dict(),
            "config": self.config.to_dict(),
            "initial_code": self.initial_code,
            "additional_requirements": self.additional_requirements,
            "local_memory": self.local_memory,
            "global_memory": self.global_memory,
            "output_dir": str(self.output_dir) if self.output_dir else None,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        instance_cls: type | None = None,
        config_cls: type | None = None,
    ) -> PerfAgentRequest:
        """从字典创建请求对象

        Args:
            data: 字典数据
            instance_cls: EffiBenchXInstance 类（延迟导入时使用）
            config_cls: PerfAgentConfig 类（延迟导入时使用）
        """
        # 延迟导入以避免循环依赖
        if instance_cls is None:
            from .agent import EffiBenchXInstance

            instance_cls = EffiBenchXInstance
        if config_cls is None:
            from .config import PerfAgentConfig

            config_cls = PerfAgentConfig

        instance = instance_cls.from_dict(data["instance"])
        config = config_cls.from_dict(data["config"])
        output_dir = Path(data["output_dir"]) if data.get("output_dir") else None

        return cls(
            instance=instance,
            config=config,
            initial_code=data.get("initial_code"),
            additional_requirements=data.get("additional_requirements"),
            local_memory=data.get("local_memory"),
            global_memory=data.get("global_memory"),
            output_dir=output_dir,
        )


@dataclass
class PerfAgentResult:
    """PerfAgent 的输出结果

    这是 PerfAgent 返回给 SE_Perf 的标准化结果对象。
    包含优化结果、性能指标以及相关元数据。

    Attributes:
        instance_id: 实例 ID
        success: 优化是否成功（性能有提升）
        initial_code: 初始代码
        optimized_code: 优化后的代码
        initial_performance: 初始性能值
        final_performance: 最终性能值
        final_metrics: 最终性能指标字典（runtime, memory, integral）
        language: 编程语言
        optimization_target: 优化目标（runtime/memory/integral）
        performance_unit: 性能单位
        total_iterations: 总迭代次数
        optimization_history: 优化历史记录
        trajectory_file: 轨迹文件路径（可选）
        final_artifacts: 最终产物描述（可选）
        error: 错误信息（如果执行失败）
    """

    instance_id: str
    success: bool
    # 代码
    initial_code: str
    optimized_code: str
    # 性能指标
    initial_performance: float
    final_performance: float
    final_metrics: dict[str, Any] = field(default_factory=dict)
    # 元数据
    language: str = "python3"
    optimization_target: str = "runtime"
    performance_unit: str = "s"
    total_iterations: int = 0
    optimization_history: list[dict[str, Any]] = field(default_factory=list)
    # 可选字段
    trajectory_file: str | None = None
    final_artifacts: str | None = None
    error: str | None = None

    @property
    def improvement_pct(self) -> float:
        """计算改进百分比"""
        if self.initial_performance <= 0 or self.initial_performance == float("inf"):
            return 0.0
        return (self.initial_performance - self.final_performance) / self.initial_performance * 100

    @property
    def passed(self) -> bool:
        """是否通过（有优化效果或至少执行成功）"""
        return self.success or self.error is None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "instance_id": self.instance_id,
            "success": self.success,
            "initial_code": self.initial_code,
            "optimized_code": self.optimized_code,
            "initial_performance": self.initial_performance,
            "final_performance": self.final_performance,
            "final_metrics": self.final_metrics,
            "language": self.language,
            "optimization_target": self.optimization_target,
            "performance_unit": self.performance_unit,
            "total_iterations": self.total_iterations,
            "optimization_history": self.optimization_history,
            "trajectory_file": self.trajectory_file,
            "final_artifacts": self.final_artifacts,
            "error": self.error,
            # 计算字段
            "improvement_pct": self.improvement_pct,
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerfAgentResult:
        """从字典创建结果对象

        兼容 PerfAgent.run() 返回的原始结果格式
        """
        return cls(
            instance_id=data.get("instance_id", "unknown"),
            success=data.get("success", False),
            initial_code=data.get("initial_code", ""),
            optimized_code=data.get("optimized_code", ""),
            initial_performance=_safe_float(data.get("initial_performance", float("inf"))),
            final_performance=_safe_float(data.get("final_performance", float("inf"))),
            final_metrics=data.get("final_metrics", {}),
            language=data.get("language", "python3"),
            optimization_target=data.get("optimization_target", "runtime"),
            performance_unit=data.get("performance_unit", "s"),
            total_iterations=data.get("total_iterations", 0),
            optimization_history=data.get("optimization_history", []),
            trajectory_file=data.get("trajectory_file"),
            final_artifacts=data.get("final_artifacts"),
            error=data.get("error"),
        )

    @classmethod
    def from_error(cls, instance_id: str, error: str) -> PerfAgentResult:
        """从错误创建失败结果"""
        return cls(
            instance_id=instance_id,
            success=False,
            initial_code="",
            optimized_code="",
            initial_performance=float("inf"),
            final_performance=float("inf"),
            error=error,
        )

    def to_json(self, path: Path | str) -> None:
        """保存为 JSON 文件"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, path: Path | str) -> PerfAgentResult:
        """从 JSON 文件加载"""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def _safe_float(value: Any) -> float:
    """安全转换为浮点数"""
    if value is None:
        return float("inf")
    if isinstance(value, str):
        if value.lower() in ("inf", "infinity"):
            return float("inf")
        if value.lower() in ("-inf", "-infinity"):
            return float("-inf")
        if value.lower() == "nan":
            return float("nan")
        try:
            return float(value)
        except ValueError:
            return float("inf")
    try:
        return float(value)
    except (ValueError, TypeError):
        return float("inf")
