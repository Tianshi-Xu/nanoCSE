#!/usr/bin/env python3
"""
测试 SE_Perf 与 PerfAgent 之间的接口协议

验证内容：
1. protocols.py 文件的语法正确性和结构
2. PerfAgentResult / PerfAgentRequest 的实例化、序列化、apply_overrides
3. perf_run.py 的单实例接口
4. Operator 的 run_for_instance 接口
5. json_utils 公共模块
"""

import ast
import math
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent

# 确保 perfagent 和 SE_Perf 可导入
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "SE_Perf"))


# ---------------------------------------------------------------------------
# 基础语法 / 结构测试
# ---------------------------------------------------------------------------


def test_protocols_syntax():
    """验证 protocols.py 的语法正确性"""
    protocols_file = project_root / "perfagent" / "protocols.py"
    assert protocols_file.exists(), f"protocols.py 文件不存在: {protocols_file}"

    code = protocols_file.read_text()
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise AssertionError(f"protocols.py 语法错误: {e}")

    class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    expected_classes = ["PerfAgentRequest", "PerfAgentResult"]
    for cls_name in expected_classes:
        assert cls_name in class_names, f"缺少类定义: {cls_name}"

    print(f"  protocols.py 语法正确，包含类: {class_names}")


def test_protocols_structure():
    """验证 protocols.py 中类的结构"""
    protocols_file = project_root / "perfagent" / "protocols.py"
    code = protocols_file.read_text()
    tree = ast.parse(code)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "PerfAgentResult":
            methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            assert "to_dict" in methods, "PerfAgentResult 缺少 to_dict 方法"
            assert "from_dict" in methods, "PerfAgentResult 缺少 from_dict 方法"
            assert "from_error" in methods, "PerfAgentResult 缺少 from_error 方法"

        if isinstance(node, ast.ClassDef) and node.name == "PerfAgentRequest":
            methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            assert "to_dict" in methods, "PerfAgentRequest 缺少 to_dict 方法"
            assert "from_dict" in methods, "PerfAgentRequest 缺少 from_dict 方法"
            assert "apply_overrides" in methods, "PerfAgentRequest 缺少 apply_overrides 方法"

    print("  protocols.py 类结构验证通过")


# ---------------------------------------------------------------------------
# PerfAgentResult 行为测试
# ---------------------------------------------------------------------------


def test_perf_agent_result_instantiation():
    """验证 PerfAgentResult 的实例化"""
    from perfagent.protocols import PerfAgentResult

    # 正常创建
    result = PerfAgentResult(
        instance_id="test-001",
        success=True,
        initial_code="# init",
        optimized_code="# opt",
        initial_performance=10.0,
        final_performance=5.0,
        final_metrics={"runtime": 5.0, "memory": 100.0, "integral": 500.0},
    )
    assert result.instance_id == "test-001"
    assert result.success is True
    assert result.improvement_pct == 50.0
    assert result.passed is True

    # from_error
    err_result = PerfAgentResult.from_error("fail-001", "boom")
    assert err_result.success is False
    assert err_result.error == "boom"
    assert err_result.initial_performance == float("inf")

    print("  PerfAgentResult 实例化通过")


def test_perf_agent_result_serialization():
    """验证 PerfAgentResult 的序列化与反序列化"""
    from perfagent.protocols import PerfAgentResult

    original = PerfAgentResult(
        instance_id="ser-001",
        success=True,
        initial_code="a = 1",
        optimized_code="a = 2",
        initial_performance=8.0,
        final_performance=3.0,
        final_metrics={"runtime": 3.0},
        language="python3",
        optimization_target="runtime",
    )

    d = original.to_dict()
    assert isinstance(d, dict)
    assert d["instance_id"] == "ser-001"
    assert d["success"] is True
    assert "improvement_pct" in d
    assert "passed" in d

    restored = PerfAgentResult.from_dict(d)
    assert restored.instance_id == original.instance_id
    assert restored.success == original.success
    assert restored.final_performance == original.final_performance
    assert restored.language == original.language

    print("  PerfAgentResult 序列化 / 反序列化通过")


# ---------------------------------------------------------------------------
# perf_run.py 单实例接口测试
# ---------------------------------------------------------------------------


def test_perf_run_single_instance_interface():
    """验证 perf_run.py 的单实例接口"""
    perf_run_file = project_root / "SE_Perf" / "perf_run.py"
    assert perf_run_file.exists(), "perf_run.py 文件不存在"

    code = perf_run_file.read_text()

    # 不再有批量调用函数
    assert "def build_requests_from_params" not in code, "perf_run.py 中不应存在 build_requests_from_params"
    assert "def call_perfagent_subprocess" not in code, "perf_run.py 中不应存在 call_perfagent_subprocess"
    assert "def create_temp_perf_config" not in code, "perf_run.py 中不应存在 create_temp_perf_config"
    assert "def _inject_global_memory" not in code, "perf_run.py 中不应存在 _inject_global_memory"

    # 应有单实例核心函数
    assert "def _run_single_perfagent" in code, "缺少 _run_single_perfagent 函数"
    assert "def _retrieve_global_memory" in code, "缺少 _retrieve_global_memory 函数"
    assert "def _build_perf_agent_config" in code, "缺少 _build_perf_agent_config 函数"
    assert "def write_iteration_preds_from_result" in code, "缺少 write_iteration_preds_from_result 函数"

    # 应使用 PerfAgentRequest / PerfAgentResult 协议
    assert "PerfAgentRequest" in code, "perf_run.py 未使用 PerfAgentRequest"
    assert "PerfAgentResult" in code, "perf_run.py 未使用 PerfAgentResult"

    # 应有 --instance 参数
    assert "--instance" in code, "perf_run.py 缺少 --instance CLI 参数"

    print("  perf_run.py 单实例接口验证通过")


# ---------------------------------------------------------------------------
# Operator run_for_instance 接口测试
# ---------------------------------------------------------------------------


def test_operator_run_for_instance():
    """验证 Operator 的 run_for_instance 接口"""
    operators_dir = project_root / "SE_Perf" / "operators"

    # 检查 base.py
    base_file = operators_dir / "base.py"
    assert base_file.exists()
    base_code = base_file.read_text()
    assert "class OperatorResult" in base_code, "base.py 缺少 OperatorResult 定义"
    assert "def run_for_instance" in base_code, "base.py 缺少 run_for_instance 定义"

    # 检查各算子实现了 run_for_instance
    for name in ["crossover.py", "reflection_refine.py", "alternative_strategy.py",
                  "trajectory_analyzer.py", "traj_pool_summary.py", "plan.py", "filter.py"]:
        op_file = operators_dir / name
        assert op_file.exists(), f"算子文件不存在: {name}"
        op_code = op_file.read_text()
        assert "def run_for_instance" in op_code, f"{name} 未实现 run_for_instance"
        assert "OperatorResult" in op_code, f"{name} 未使用 OperatorResult"

    # plan.py 应返回 list[OperatorResult]
    plan_code = (operators_dir / "plan.py").read_text()
    assert "list[OperatorResult]" in plan_code, "plan.py 的 run_for_instance 应返回 list[OperatorResult]"

    print("  Operator run_for_instance 接口验证通过")


# ---------------------------------------------------------------------------
# json_utils 测试
# ---------------------------------------------------------------------------


def test_json_utils():
    """验证 json_utils 公共模块"""
    from perfagent.utils.json_utils import json_safe

    # 基础类型
    assert json_safe(None) is None
    assert json_safe(42) == 42
    assert json_safe("hello") == "hello"
    assert json_safe(True) is True

    # 浮点边界
    assert json_safe(3.14) == 3.14
    assert json_safe(float("inf")) == "Infinity"
    assert json_safe(float("-inf")) == "-Infinity"
    assert json_safe(float("nan")) == "NaN"

    # 容器
    assert json_safe({"a": 1, "b": float("inf")}) == {"a": 1, "b": "Infinity"}
    assert json_safe([1, None, "x"]) == [1, None, "x"]
    assert json_safe((1, 2)) == [1, 2]

    # Path
    assert json_safe(Path("/tmp/test")) == "/tmp/test"

    print("  json_utils 公共模块验证通过")


def test_agent_run_with_request():
    """验证 agent.py 的 run_with_request 接口"""
    agent_file = project_root / "perfagent" / "agent.py"
    assert agent_file.exists()
    code = agent_file.read_text()

    assert "def run_with_request" in code, "缺少 run_with_request 方法"

    print("  agent.py run_with_request 接口验证通过")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("运行 SE_Perf 与 PerfAgent 接口协议测试")
    print("=" * 60 + "\n")

    try:
        test_protocols_syntax()
        test_protocols_structure()
        test_perf_agent_result_instantiation()
        test_perf_agent_result_serialization()
        test_perf_run_single_instance_interface()
        test_operator_run_for_instance()
        test_json_utils()
        test_agent_run_with_request()

        print("\n" + "=" * 60)
        print("所有测试通过！")
        print("=" * 60)
        print("\n重构完成的内容：")
        print("  1. perfagent/protocols.py - 标准化数据接口")
        print("  2. perfagent/agent.py - run_with_request() 标准化 API")
        print("  3. perfagent/utils/json_utils.py - 公共 JSON 序列化工具")
        print("  4. SE_Perf/perf_run.py - 单实例模式，直接构建 PerfAgentRequest")
        print("  5. SE_Perf/operators/*.py - run_for_instance 返回 OperatorResult")
        print("  6. 删除: YAML 文件中间通道、subprocess 调用、批量处理逻辑")
        print()
        return True

    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
