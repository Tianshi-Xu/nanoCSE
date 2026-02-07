from unittest.mock import patch

import pytest

from perfagent.agent import EffiBenchXInstance, PerfAgent
from perfagent.config import PerfAgentConfig


class TestRunRefactored:
    @pytest.fixture
    def mock_llm_client(self):
        with patch("perfagent.agent.LLMClient") as mock:
            client_instance = mock.return_value
            # Configure call_llm to return a diff that replaces the placeholder code
            client_instance.call_llm.return_value = """
<<<<<<< SEARCH
# Start your code here
=======
def solve():
    return 42
>>>>>>> REPLACE
"""
            yield client_instance

    @pytest.fixture
    def mock_benchmark(self):
        with patch("perfagent.tasks.effibench.run_performance_benchmark") as mock:
            # We will set side_effect in individual tests
            yield mock

    @pytest.fixture
    def config(self, tmp_path):
        from conftest import TEST_OPTIMIZATION_TEMPLATE, TEST_SYSTEM_TEMPLATE

        from perfagent.config import PromptConfig

        cfg = PerfAgentConfig(
            max_iterations=1,
            task_config={
                "language": "python3",
                "num_runs": 5,
            },
            prompts=PromptConfig(
                system_template=TEST_SYSTEM_TEMPLATE,
                optimization_template=TEST_OPTIMIZATION_TEMPLATE,
            ),
        )
        cfg.logging.log_dir = tmp_path / "logs"
        cfg.logging.trajectory_dir = tmp_path / "traj"
        cfg.model.api_key = "dummy"  # Trigger LLM client init
        cfg.model.api_base = "http://dummy"
        return cfg

    @pytest.fixture
    def agent(self, config, mock_llm_client):
        return PerfAgent(config)

    @pytest.fixture
    def instance(self):
        return EffiBenchXInstance(
            id="test-inst-1",
            title="Test Instance",
            title_slug="test-instance",
            description="A test problem",
            description_md="# Test Problem",
            source="leetcode",
            url="http://example.com",
            type="functional",
            starter_code={"python3": "def solve():\n    pass"},
            generated_tests=[{"input": "1", "output": "1"}],
            evaluator="def check(): pass",  # Added evaluator
        )

    def test_run_success_flow(self, agent, instance, mock_llm_client, mock_benchmark):
        """Test the full run flow with mocked dependencies.

        New flow: no initial evaluation, just optimization loop.
        With max_iterations=1:
        - 1 iteration: single run + full run = 2 benchmark calls
        """

        # Setup benchmark side effects (2 calls for 1 iteration)
        mock_benchmark.side_effect = [
            # Iteration 1 - Single Run (Pass)
            {
                "performance_analysis": {"runtime": 0.5, "memory": 100, "passed": True, "pass_rate": 1.0},
                "pass_rates": [1.0],
                "failed_test_details": [],
                "first_run_details": [{"passed": True}],
            },
            # Iteration 1 - Full Run (Pass, Runtime 0.5)
            {
                "performance_analysis": {"runtime": 0.5, "memory": 100, "passed": True, "pass_rate": 1.0},
                "pass_rates": [1.0],
                "failed_test_details": [],
                "first_run_details": [{"passed": True}],
            },
        ]

        result = agent.run(instance)

        # Verify result structure
        assert result["success"] is True
        assert result["total_iterations"] >= 1
        assert result["metric"] == 0.5

        # Verify LLM was called
        mock_llm_client.call_llm.assert_called()

        # Verify benchmark was called (2 calls for 1 iteration: single + full)
        assert mock_benchmark.call_count == 2

    def test_run_no_improvement(self, agent, instance, mock_llm_client, mock_benchmark):
        """Test flow where optimization does not improve over placeholder."""
        # Set max_iterations > 1 so we don't force save
        agent.config.max_iterations = 2
        agent.config.early_stop_no_improve = 1  # Stop after 1 failed iter

        # Mock benchmark returns failed tests (pass_rate < 1.0)
        mock_benchmark.side_effect = [
            # Iteration 1 - Single Run (Fail)
            {
                "performance_analysis": {"runtime": 1.5, "passed": False, "pass_rate": 0.5},
                "pass_rates": [0.5],
                "failed_test_details": [{"status": "wrong_answer"}],
                "first_run_details": [{"passed": False}],
            },
        ]

        result = agent.run(instance)

        # No valid metric means not successful
        assert result["success"] is False

    def test_run_single_iteration_forced_save(self, agent, instance, mock_llm_client, mock_benchmark):
        """Test that single iteration run saves generated code even if metric is inf."""
        agent.config.max_iterations = 1

        # Mock benchmark: passes but with some metric
        mock_benchmark.side_effect = [
            # Iteration 1 - Single Run (Pass)
            {
                "performance_analysis": {"runtime": 2.0, "passed": True, "pass_rate": 1.0},
                "pass_rates": [1.0],
                "failed_test_details": [],
                "first_run_details": [{"passed": True}],
            },
            # Iteration 1 - Full Run (Pass, Runtime 2.0)
            {
                "performance_analysis": {"runtime": 2.0, "passed": True, "pass_rate": 1.0},
                "pass_rates": [1.0],
                "failed_test_details": [],
                "first_run_details": [{"passed": True}],
            },
        ]

        result = agent.run(instance)

        assert result["total_iterations"] == 1
        assert result["metric"] == 2.0
        # With single iteration mode, the result is force-saved
        assert result["success"] is True
        assert len(result["artifacts"]["optimization_history"]) == 1
        assert result["artifacts"]["optimization_history"][0]["success"] is True  # Forced success flag
