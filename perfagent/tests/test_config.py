"""
配置系统的 pytest 测试用例
"""

import tempfile
from pathlib import Path

import pytest

from perfagent.config import ModelConfig, PerfAgentConfig


class TestPerfAgentConfig:
    """测试配置系统"""

    def test_default_config(self):
        """测试默认配置"""
        config = PerfAgentConfig()
        assert config.max_iterations == 10
        assert config.model.name == "gpt-4"
        assert config.logging.save_trajectory is True
        assert config.adopt_only_if_improved is False
        assert config.task_config == {}

    def test_yaml_config(self):
        """测试 YAML 配置加载"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_content = """
max_iterations: 5
model:
  name: "gpt-3.5-turbo"
  temperature: 0.2
task_config:
  target: "memory"
  language: "python3"
"""
            f.write(yaml_content)
            f.flush()

            config = PerfAgentConfig.from_yaml(Path(f.name))
            assert config.max_iterations == 5
            assert config.model.name == "gpt-3.5-turbo"
            assert config.model.temperature == 0.2
            assert config.task_config["target"] == "memory"
            assert config.task_config["language"] == "python3"

    def test_yaml_backward_compat(self):
        """测试旧格式 YAML 的向后兼容"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_content = """
max_iterations: 3
optimization:
  target: "runtime"
  code_generation_mode: "diff"
  adopt_only_if_improved: true
runtime:
  time_limit: 30
  num_runs: 5
language_cfg:
  language: "cpp"
"""
            f.write(yaml_content)
            f.flush()

            config = PerfAgentConfig.from_yaml(Path(f.name))
            assert config.max_iterations == 3
            assert config.adopt_only_if_improved is True
            # 旧格式的字段应迁移到 task_config
            assert config.task_config["target"] == "runtime"
            assert config.task_config["code_generation_mode"] == "diff"
            assert config.task_config["time_limit"] == 30
            assert config.task_config["num_runs"] == 5
            assert config.task_config["language"] == "cpp"

    def test_config_validation(self):
        """测试配置验证"""
        # 测试无效的最大迭代次数
        with pytest.raises(ValueError):
            PerfAgentConfig(max_iterations=-1)

        # 测试无效的温度值
        with pytest.raises(ValueError):
            PerfAgentConfig(model=ModelConfig(temperature=2.0))

    def test_config_serialization(self):
        """测试配置序列化"""
        config = PerfAgentConfig(
            max_iterations=5,
            model=ModelConfig(name="gpt-3.5-turbo", temperature=0.3),
            task_config={"target": "runtime", "language": "python3"},
        )

        # 测试转换为字典（嵌套结构）
        config_dict = config.to_dict()
        assert config_dict["max_iterations"] == 5
        assert config_dict["model"]["name"] == "gpt-3.5-turbo"
        assert config_dict["model"]["temperature"] == 0.3
        assert config_dict["task_config"]["target"] == "runtime"

        # 测试从字典创建（嵌套结构）
        new_config = PerfAgentConfig.from_dict(config_dict)
        assert new_config.max_iterations == config.max_iterations
        assert new_config.model.name == config.model.name
        assert new_config.model.temperature == config.model.temperature
        assert new_config.task_config == config.task_config

    def test_task_config_preserved_in_yaml_roundtrip(self):
        """测试 task_config 在 YAML 序列化/反序列化中保持不变"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_content = """
max_iterations: 2
adopt_only_if_improved: true
task_config:
  target: "integral"
  code_generation_mode: "direct"
  language: "python3"
  time_limit: 10
  num_runs: 5
"""
            f.write(yaml_content)
            f.flush()

            config = PerfAgentConfig.from_yaml(Path(f.name))
            assert config.adopt_only_if_improved is True
            assert config.task_config["target"] == "integral"
            assert config.task_config["code_generation_mode"] == "direct"
            assert config.task_config["time_limit"] == 10
