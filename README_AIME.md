# AIME 任务支持实现总结

本文档总结了在 `nanoCSE` 框架中支持 AIME（数学推理）任务所做的更改。

## 代码实现
- **`perfagent/tasks/aime.py`**:
  实现了 `AIMERunner` 类。它管理 AIME 数学问题的执行，并使用精确匹配的二元评分机制（-1.0/1.0）。
- **`perfagent/tasks/__init__.py`**:
  更新以导出 `AIMERunner` 并在系统中注册 `aime` 任务类型。

## 配置文件
- **`configs/aime_evolve.yaml`**:
  运行 AIME 问题进化优化的主要入口配置。指定了模型（例如 `step-3.5-flash:free`）、迭代参数以及关联的性能配置。
- **`configs/perf_configs/config_aime_evolve.yaml`**:
  AIME 的详细性能配置，实现了 `metric_higher_is_better: true` 逻辑并设置了评估超时。

## 数据处理与工具
- **`utils/convert_aime_parquet.py`**:
  用于将外部来源的 AIME parquet 数据集（来自 `Open-AgentRL-Eval`）转换为系统原生 JSON 格式的工具脚本。处理了字段大小写标准化（"Problem" -> "problem"）和模式清理（剔除 "url"）。
- **`instances/aime2024_converted/` & `instances/aime2025_converted/`**:
  生成的目录，包含已转换好的 AIME 2024 和 2025 实例，可直接使用。
