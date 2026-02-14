"""
AIME TaskRunner 实现（数学推理）

基于 LiveCodeBench TaskRunner 的结构，使用 math_dapo.compute_score
进行正确性判分。面向 AIME/数学题场景，输出为 Answer: <number>。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from perfagent.task_runner import BaseTaskRunner
from perfagent.protocols import TaskMetadata
from perfagent.tasks.math_dapo import compute_score, last_boxed_only_string, remove_boxed


@dataclass
class AIMEConfig:
    """AIME 任务特定配置"""

    strict_box_verify: bool = False

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any] | None) -> "AIMEConfig":
        if config_dict is None:
            return cls()
        return cls(strict_box_verify=bool(config_dict.get("strict_box_verify", False)))


@dataclass
class AIMEInstance:
    """AIME 实例数据（尽量兼容不同数据源）"""

    id: str
    problem: str
    answer: str | None
    metadata: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any], file_path: Path | None = None) -> "AIMEInstance":
        instance_id = (
            data.get("id")
            or data.get("ID")
            or data.get("instance_id")
            or data.get("problem_id")
            or data.get("question_id")
        )
        if not instance_id and file_path:
            instance_id = file_path.stem
        if not instance_id:
            instance_id = "unknown"

        problem = (
            data.get("problem")
            or data.get("Problem")
            or data.get("question")
            or data.get("question_content")
            or data.get("prompt")
            or data.get("problem_description")
            or data.get("statement")
            or ""
        )

        answer = (
            data.get("answer")
            or data.get("Answer")
            or data.get("final_answer")
            or data.get("ground_truth")
            or data.get("solution")
            or data.get("target")
        )
        if answer is not None:
            answer = str(answer)

        metadata = dict(data.get("metadata", {}))
        return cls(id=str(instance_id), problem=str(problem), answer=answer, metadata=metadata)


class AIMERunner(BaseTaskRunner):
    """AIME 数学推理任务的 TaskRunner 实现"""

    def __init__(self, *, task_config: dict[str, Any] | None = None, _logger: logging.Logger | None = None):
        self._logger = _logger or logging.getLogger(__name__)
        self._task_config = AIMEConfig.from_dict(task_config)

    # ------------------------------------------------------------------
    # 元数据提取（类方法，无需实例化）
    # ------------------------------------------------------------------

    @classmethod
    def load_metadata(cls, path: Path) -> TaskMetadata:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.load_metadata_from_dict(data, file_path=path)

    @classmethod
    def load_metadata_from_dict(cls, data: dict[str, Any], file_path: Path | None = None) -> TaskMetadata:
        instance = AIMEInstance.from_dict(data, file_path)
        return TaskMetadata(
            instance_id=instance.id,
            problem_description=instance.problem,
        )

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    def load_instance(self, path: Path) -> AIMEInstance:
        data = json.loads(path.read_text(encoding="utf-8"))
        return self.load_instance_from_dict(data, file_path=path)

    def load_instance_from_dict(self, data: dict[str, Any], file_path: Path | None = None) -> AIMEInstance:
        return AIMEInstance.from_dict(data, file_path)

    # ------------------------------------------------------------------
    # 初始解
    # ------------------------------------------------------------------

    def get_initial_solution(self, instance_data: Any, config: Any) -> str:
        """AIME 初始解为空字符串，直接由模型生成答案"""
        return ""

    # ------------------------------------------------------------------
    # 评估
    # ------------------------------------------------------------------

    def evaluate(self, solution: str, instance_data: Any, config: Any) -> tuple[float, dict[str, Any]]:
        """评估数学题解的正确性

        使用 compute_score，返回 score（1.0 或 -1.0）。
        """
        instance: AIMEInstance = instance_data
        if not instance.answer:
            self._logger.warning(f"实例 {instance.id} 缺少标准答案，无法评估")
            return -1.0, {
                "acc": False,
                "pred": None,
                "score": -1.0,
                "evaluation_metadata": {"error": "missing_ground_truth"},
            }

        score_dict = compute_score(
            solution_str=solution,
            ground_truth=instance.answer,
            strict_box_verify=self._task_config.strict_box_verify,
        )

        metric = float(score_dict.get("score", -1.0))
        artifacts: dict[str, Any] = {
            "acc": bool(score_dict.get("acc", False)),
            "pred": score_dict.get("pred"),
            "score": metric,
        }

        self._logger.info(
            f"实例 {instance.id}: {'正确' if artifacts['acc'] else '错误'}"
        )

        return metric, artifacts

    # ------------------------------------------------------------------
    # Prompt 构建
    # ------------------------------------------------------------------

    def build_system_prompt(self, instance_data: Any, **context: Any) -> str:
        """构建系统 prompt

        Expected context keys:
            config: PerfAgentConfig 对象
            additional_requirements: str（可选，优先于 config）
            local_memory: str（可选）
            global_memory: str（可选）
        """
        instance: AIMEInstance = instance_data
        config = context.get("config")

        if not config or not hasattr(config, "prompts"):
            raise ValueError("config.prompts 未配置，无法构建 system prompt")

        tmpl = getattr(config.prompts, "system_template", "")
        if not tmpl:
            raise ValueError("config.prompts.system_template 为空，无法构建 system prompt")

        additional_requirements = (
            context.get("additional_requirements")
            or getattr(getattr(config, "prompts", None), "additional_requirements", None)
            or ""
        )
        local_memory = (
            context.get("local_memory")
            or getattr(getattr(config, "prompts", None), "local_memory", None)
            or ""
        )
        global_memory = (
            context.get("global_memory")
            or getattr(getattr(config, "prompts", None), "global_memory", None)
            or ""
        )

        try:
            return tmpl.format(
                problem_statement=instance.problem,
                additional_requirements=additional_requirements,
                local_memory=local_memory,
                global_memory=global_memory,
            )
        except KeyError as e:
            raise ValueError(f"system_prompt 模板填充失败，缺少占位符: {e}")

    def build_optimization_prompt(
        self,
        solution: str,
        metric: float,
        artifacts: dict[str, Any],
        **context: Any,
    ) -> str:
        """构建优化 prompt"""
        config = context.get("config")

        if not config or not hasattr(config, "prompts"):
            raise ValueError("config.prompts 未配置，无法构建 optimization prompt")

        tmpl = getattr(config.prompts, "optimization_template", "")
        if not tmpl:
            raise ValueError("config.prompts.optimization_template 为空，无法构建 optimization prompt")

        correct = artifacts.get("acc", False)
        pred = artifacts.get("pred", None)

        try:
            return tmpl.format(
                current_solution=solution,
                metric=metric,
                correct="正确" if correct else "错误",
                pred=pred if pred is not None else "",
            )
        except KeyError as e:
            raise ValueError(f"optimization_prompt 模板填充失败，缺少占位符: {e}")

    # ------------------------------------------------------------------
    # 解提取
    # ------------------------------------------------------------------

    def extract_solution(self, llm_response: str, current_solution: str) -> str:
        """从 LLM 响应中提取最终答案

        优先匹配：
        1. \\boxed{...} (AIME 标准格式，且 Prompt 明确要求)
        2. Answer: xxx
        若都无法提取，则返回原响应。
        """
        # 1. 优先尝试提取 \\boxed{...} (最准确)
        boxed = last_boxed_only_string(llm_response)
        if boxed:
            try:
                unboxed = remove_boxed(boxed)
                return f"Answer: {unboxed}"
            except Exception:
                return f"Answer: {boxed}"

        # 2. 尝试提取 Answer: xxx
        answer_matches = re.findall(r"(?i)Answer\s*:\s*([^\n]+)", llm_response)
        if answer_matches:
            answer = answer_matches[-1].strip()
            # 简单验证：不能为空或仅含 $$
            if answer and answer != "$$":
                return f"Answer: {answer}"

        return llm_response.strip() if llm_response.strip() else current_solution
