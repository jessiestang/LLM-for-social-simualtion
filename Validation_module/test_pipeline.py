# test_validation_pipeline.py

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from validator import ModelValidation

# ─────────────────────────────────────────────
# MOCK DATA
# ─────────────────────────────────────────────

MOCK_CONCEPTUAL_MODEL = {
    "model_title": "Spiral of Silence",
    "overview": "Simulates opinion suppression in social networks",
    "agents": {
        "types": ["HumanAgent", "LLMAgent"],
        "attributes": {
            "HumanAgent": [{"opinion": "float", "is_speaking": "boolean"}],
            "LLMAgent":   [{"opinion": "float", "is_speaking": "boolean"}]
        }
    },
    "simulation_parameters": {
        "num_agents": 100,
        "time_steps": 50,
        "constants": {
            "beta": "balances global vs local opinion",
            "default_value_beta": 0.5,
            "range_beta": "0.0 to 1.0"
        }
    }
}

MOCK_EVALUATION_SUGGESTIONS = [
    {
        "strategy_id": "1",
        "strategy_type": "Stochasticity Control",
        "description": "Run simulation 30 times with different random seeds."
    },
    {
        "strategy_id": "2",
        "strategy_type": "Parameter Sensitivity Analysis",
        "description": "Vary beta from 0.0 to 1.0 in increments of 0.1."
    },
    {
        "strategy_id": "3",
        "strategy_type": "Uncertainty Quantification",
        "description": "Compute 95% confidence interval using bootstrap resampling."
    },
    {
        "strategy_id": "4",
        "strategy_type": "Cross-condition Comparison",
        "description": "Compare low/medium/high beta conditions using ANOVA."
    }
]

MOCK_MODEL_INTERFACE = {
    "model_class": "SpiralSilenceModel",
    "init_params": ["num_agents", "beta", "seed"],
    "run_method": "run_model(steps)",
    "data_collector": "datacollector.get_model_vars_dataframe()"
}

MOCK_EVALUATION_CODE = """
import pandas as pd
from model import SpiralSilenceModel

results = []
for seed in range(30):
    model = SpiralSilenceModel(num_agents=100, beta=0.5, seed=seed)
    model.run_model(steps=50)
    df = model.datacollector.get_model_vars_dataframe()
    results.append(df)
"""

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def make_mock_llm_context(response_content: str):
    """Create a mock llm_context that returns a fixed response."""
    mock_context = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = response_content
    mock_context.chat.return_value = mock_response
    return mock_context


def make_agent(mock_response: str):
    """Instantiate ModelValidation with mocked LLM and file I/O."""
    agent = ModelValidation.__new__(ModelValidation)
    agent.llm_context = make_mock_llm_context(mock_response)
    agent.model_name = "gpt-4o-mini"
    return agent


# ─────────────────────────────────────────────
# TESTS — evaluation_suggestion
# ─────────────────────────────────────────────

def test_evaluation_suggestion_valid_json():
    """LLM returns clean JSON — should parse correctly."""
    agent = make_agent(json.dumps(MOCK_EVALUATION_SUGGESTIONS))

    with patch("builtins.open", mock_open(read_data=json.dumps(MOCK_CONCEPTUAL_MODEL))):
        result = agent.evaluation_suggestion("fake/path/model.json")

    assert isinstance(result, list),            f"Expected list, got {type(result)}"
    assert len(result) == 4,                    f"Expected 4 strategies, got {len(result)}"
    assert result[0]["strategy_id"] == "1"
    assert "strategy_type" in result[0]
    assert "description"   in result[0]
    print("✅ evaluation_suggestion: valid JSON passed")


def test_evaluation_suggestion_with_trailing_comma():
    """LLM returns JSON with trailing comma — _safe_json_load should handle it."""
    raw = """
    [
        {"strategy_id": "1", "strategy_type": "Stochasticity Control", "description": "..."},
        {"strategy_id": "2", "strategy_type": "Parameter Sensitivity Analysis", "description": "..."}
    ]
    """
    agent = make_agent(raw)

    with patch("builtins.open", mock_open(read_data=json.dumps(MOCK_CONCEPTUAL_MODEL))):
        result = agent.evaluation_suggestion("fake/path/model.json")

    assert isinstance(result, list)
    assert len(result) == 2
    print("✅ evaluation_suggestion: trailing comma handled")


def test_evaluation_suggestion_with_markdown_fences():
    """LLM wraps output in ```json fences — should still parse."""
    raw = f"```json\n{json.dumps(MOCK_EVALUATION_SUGGESTIONS)}\n```"
    agent = make_agent(raw)

    with patch("builtins.open", mock_open(read_data=json.dumps(MOCK_CONCEPTUAL_MODEL))):
        result = agent.evaluation_suggestion("fake/path/model.json")

    assert isinstance(result, list)
    print("✅ evaluation_suggestion: markdown fences handled")


def test_evaluation_suggestion_file_not_found():
    """Model file doesn't exist — should return clear error."""
    agent = make_agent(json.dumps(MOCK_EVALUATION_SUGGESTIONS))

    result = agent.evaluation_suggestion("E:\LLM_for_abm\LLM-for-social-simualtion\model_construct_assistant\_spiral_silence_model2.json")

    assert "Error" in str(result), "Should return error for missing file"
    print("✅ evaluation_suggestion: missing file handled")


def test_evaluation_suggestion_all_strategies_present():
    """All 4 strategy types must be present in output."""
    agent = make_agent(json.dumps(MOCK_EVALUATION_SUGGESTIONS))
    expected_types = {
        "Stochasticity Control",
        "Parameter Sensitivity Analysis",
        "Uncertainty Quantification",
        "Cross-condition Comparison"
    }

    with patch("builtins.open", mock_open(read_data=json.dumps(MOCK_CONCEPTUAL_MODEL))):
        result = agent.evaluation_suggestion("fake/path/model.json")

    returned_types = {s["strategy_type"] for s in result}
    missing = expected_types - returned_types
    assert not missing, f"Missing strategy types: {missing}"
    print("✅ evaluation_suggestion: all strategy types present")


# ─────────────────────────────────────────────
# TESTS — evaluation_code_generator
# ─────────────────────────────────────────────

def test_evaluation_code_generator_dict_input(tmp_path):
    """Accepts dict input for evaluation_suggestions."""
    agent = make_agent(MOCK_EVALUATION_CODE)
    output_path = str(tmp_path / "eval_code.py")

    with patch("builtins.open", mock_open(read_data=json.dumps(MOCK_MODEL_INTERFACE))):
        result = agent.evaluation_code_generator(
            evaluation_suggestions=MOCK_EVALUATION_SUGGESTIONS,
            model_interface_path="fake/interface.json",
            evaluation_output_path=output_path
        )

    assert isinstance(result, str)
    assert "SpiralSilenceModel" in result
    print("✅ evaluation_code_generator: dict input passed")


def test_evaluation_code_generator_string_input(tmp_path):
    """Accepts JSON string input for evaluation_suggestions."""
    agent = make_agent(MOCK_EVALUATION_CODE)
    output_path = str(tmp_path / "eval_code.py")

    with patch("builtins.open", mock_open(read_data=json.dumps(MOCK_MODEL_INTERFACE))):
        result = agent.evaluation_code_generator(
            evaluation_suggestions=json.dumps(MOCK_EVALUATION_SUGGESTIONS),
            model_interface_path="fake/interface.json",
            evaluation_output_path=output_path
        )

    assert isinstance(result, str)
    print("✅ evaluation_code_generator: string input passed")


def test_evaluation_code_generator_strips_markdown(tmp_path):
    """Markdown fences are stripped from generated code."""
    raw_with_fences = f"```python\n{MOCK_EVALUATION_CODE}\n```"
    agent = make_agent(raw_with_fences)
    output_path = str(tmp_path / "eval_code.py")

    with patch("builtins.open", mock_open(read_data=json.dumps(MOCK_MODEL_INTERFACE))):
        result = agent.evaluation_code_generator(
            evaluation_suggestions=MOCK_EVALUATION_SUGGESTIONS,
            model_interface_path="fake/interface.json",
            evaluation_output_path=output_path
        )

    assert "```" not in result, "Markdown fences should be stripped"
    print("✅ evaluation_code_generator: markdown stripping passed")


def test_evaluation_code_generator_saves_file(tmp_path):
    """Output code is actually written to disk."""
    agent = make_agent(MOCK_EVALUATION_CODE)
    output_path = tmp_path / "eval_code.py"

    # Use real file I/O for this test
    with patch("builtins.open", mock_open(read_data=json.dumps(MOCK_MODEL_INTERFACE))):
        agent.evaluation_code_generator(
            evaluation_suggestions=MOCK_EVALUATION_SUGGESTIONS,
            model_interface_path="fake/interface.json",
            evaluation_output_path=str(output_path)
        )

    # Verify file was written
    assert output_path.exists(), "Output file should exist"
    content = output_path.read_text()
    assert len(content) > 0, "Output file should not be empty"
    print(f"✅ evaluation_code_generator: file saved to {output_path}")


# ─────────────────────────────────────────────
# RUN ALL
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Running validation module tests...\n")

    # evaluation_suggestion tests
    test_evaluation_suggestion_valid_json()
    test_evaluation_suggestion_with_trailing_comma()
    test_evaluation_suggestion_with_markdown_fences()
    test_evaluation_suggestion_file_not_found()
    test_evaluation_suggestion_all_strategies_present()

    # evaluation_code_generator tests
    test_evaluation_code_generator_dict_input(Path("test_outputs"))
    test_evaluation_code_generator_string_input(Path("test_outputs"))
    test_evaluation_code_generator_strips_markdown(Path("test_outputs"))
    test_evaluation_code_generator_saves_file(Path("test_outputs"))

    print("\n✅ All validation tests passed — safe to run with real LLM")