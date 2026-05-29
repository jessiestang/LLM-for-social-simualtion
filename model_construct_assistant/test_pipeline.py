# test_pipeline.py

import json
from unittest.mock import MagicMock, patch
from construction_agent import ModelConstructor  # replace with your actual import

# ── Fake LLM responses ──
MOCK_VARIABLES = {
    "potential_variables": [
        {"name": "perceived_public_opinion", "data_type": "float", 
         "default_value": "0.5", "update_rule": "updated each timestep"}
    ]
}

MOCK_DECISION_RULES = {
    "selected_variables": [],
    "composite_variables": [],
    "decision_rules": [
        {
            "behavioral_decision": "speak or remain silent",
            "outcome_variable": "agent.is_speaking",
            "signal_computation": [],
            "rule_pseudocode": "IF signal > threshold:\n    agent.is_speaking = True\nELSE:\n    agent.is_speaking = False",
            "parameters": []
        }
    ]
}

MOCK_MECHANISTIC_MODEL = {
    "model_title": "Test Model",
    "overview": "Test overview",
    "agents": {"types": ["HumanAgent"], "attributes": {}, "actions": {}},
    "environment": {"structure": "grid", "interaction_rules": "...", "parameters": {}},
    "model_level_mechanisms": [],
    "decision_rules": [],
    "simulation_parameters": {"num_agents": 100, "time_steps": 50, "constants": {}},
    "simulation_schedule": ["Step 1: test"]
}

def make_mock_client(response_dict: dict):
    """Create a mock OpenAI client that returns a fixed JSON response."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(response_dict)
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def test_decision_rule_variables():
    agent = ModelConstructor.__new__(ModelConstructor)
    agent.client = make_mock_client(MOCK_VARIABLES)
    agent.model_name = "gpt-4o-mini"

    result = agent.decision_rule_variables("spiral_silence.txt")

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "potential_variables" in result
    print("✅ decision_rule_variables passed")
    return result


def test_decision_rule_designer(variables):
    agent = ModelConstructor.__new__(ModelConstructor)
    agent.client = make_mock_client(MOCK_DECISION_RULES)
    agent.model_name = "gpt-4o-mini"

    # Test _ensure_dict handles both dict and string input
    result_from_dict   = agent.decision_rule_designer("spiral_silence.txt", variables)
    result_from_string = agent.decision_rule_designer("spiral_silence.txt", json.dumps(variables))

    assert isinstance(result_from_dict, dict),   "Failed with dict input"
    assert isinstance(result_from_string, dict), "Failed with string input"
    print("✅ decision_rule_designer passed (dict + string input)")
    return result_from_dict


def test_mechanism_translation(variables, decision_rules):
    agent = ModelConstructor.__new__(ModelConstructor)
    agent.client = make_mock_client(MOCK_MECHANISTIC_MODEL)
    agent.model_name = "gpt-4o-mini"

    result = agent.mechanism_translation("spiral_silence.txt", variables, decision_rules)

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "model_title" in result
    print("✅ mechanism_translation passed")
    return result


def test_save_mechanistic_model(mechanistic_model):
    agent = ModelConstructor.__new__(ModelConstructor)

    # Test 1: dict input
    r1 = agent.save_mechanistic_model(mechanistic_model, "test_output/test_model.json")
    assert "successfully" in r1, f"Save failed: {r1}"

    # Test 2: string input
    r2 = agent.save_mechanistic_model(json.dumps(mechanistic_model), "test_output/test_model2.json")
    assert "successfully" in r2, f"Save failed with string input: {r2}"

    # Test 3: invalid input
    r3 = agent.save_mechanistic_model("not valid json {{{{", "test_output/test_model3.json")
    assert "Error" in r3, "Should have returned an error for invalid input"

    print("✅ save_mechanistic_model passed (dict + string + invalid input)")


if __name__ == "__main__":
    print("Running pipeline tests...\n")
    variables      = test_decision_rule_variables()
    decision_rules = test_decision_rule_designer(variables)
    mechanistic    = test_mechanism_translation(variables, decision_rules)
    test_save_mechanistic_model(mechanistic)
    print("\n✅ All tests passed — safe to run with real LLM")