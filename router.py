from typing import Dict, Any, List
import json
from urllib import response
from openai import OpenAI
from model_construct_assistant.construction_agent import ModelConstructor
from code_generator.coding_agent import CodingAgent
from Validation_module.validator import ModelValidation


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "decision_rule_variables",
            "description": "Brainstorm key decision variables from the problem context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "problem_context": {"type": "string"}
                },
                "required": ["problem_context"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "decision_rule_designer",
            "description": "Generate executable if-then decision rules using selected variables.",
            "parameters": {
                "type": "object",
                "properties": {
                    "problem_context":   {"type": "string"},
                    "variables":         {"type": "string", "description": "JSON string of variables"},
                    "user_requirement":  {"type": "string"}
                },
                "required": ["problem_context", "variables"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "mechanism_translation",
            "description": "Convert context + variables + rules into a mechanistic conceptual model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "problem_context": {"type": "string"},
                    "variables":       {"type": "string"},
                    "decision_rules":  {"type": "string"}
                },
                "required": ["problem_context", "variables", "decision_rules"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ODD_formatter",
            "description": "Format a mechanistic model into ODD protocol text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "problem_context":   {"type": "string"},
                    "mechanistic_model": {"type": "string"}
                },
                "required": ["problem_context", "mechanistic_model"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_odd_to_wordfile",
            "description": "Save ODD text into a Word file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text":      {"type": "string"},
                    "file_path": {"type": "string"}
                },
                "required": ["text", "file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_pipeline",
            "description": "Generate/debug MESA code from a conceptual model JSON.",
            "parameters": {
                "type": "object",
                "properties": {
                    "json_path":         {"type": "string"},
                    "output_path":       {"type": "string"},
                    "user_requirements": {"type": "string"}
                },
                "required": ["json_path", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "evaluation_suggestion",
            "description": "Suggest VVUQ evaluation strategies for a conceptual model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "conceptual_model": {"type": "string"}
                },
                "required": ["conceptual_model"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "evaluation_code_generator",
            "description": "Generate evaluation code aligned with the model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_code":            {"type": "string"},
                    "evaluation_suggestions":{"type": "string"},
                    "output_path":           {"type": "string"},
                    "model_interface":       {"type": "string"}
                },
                "required": ["model_code", "evaluation_suggestions", "output_path"]
            }
        }
    },
]

ROUTER_SYSTEM_PROMPT = """
        You are a planning assistant for an LLM-assisted ABM framework.

        You DO NOT generate models, code, or validation results.
        You DO NOT execute anything.
        Your jobs are:
        Present the user with a short menu of runnable functions and explain them briefly;
        When a user asks to do something, call the appropriate function(s) using the workspace values available.
        If a required input is missing from the workspace, tell the user what they need to provide first.
        You may call multiple functions in sequence if needed.

        Available functions (callable by the program):
        A) Model construction assistant
        1) decision_rule_variables(problem_context)
        - Purpose: extract/brainstorm key decision variables from the problem context.
        - Produces: variables (json)
        - Requires: problem_context

        2) decision_rule_designer(problem_context, variables, user_requirement)
        - Purpose: generate executable if–then decision rules using selected variables.
        - Produces: decision_rules (json)
        - Requires: problem_context, variables
        - Optional: user_requirement

        3) mechanism_translation(problem_context, variables, decision_rules)
        - Purpose: convert context + variables + rules into a mechanistic conceptual model.
        - Produces: mechanistic_model (json or text)
        - Requires: problem_context, variables, decision_rules

        4) ODD_formatter(problem_context, mechanistic_model)
        - Purpose: format a mechanistic model into ODD text.
        - Produces: odd_text (string)
        - Requires: problem_context, mechanistic_model

        5) save_odd_to_wordfile(text, file_path)
        - Purpose: save ODD text into a Word file.
        - Produces: odd_docx_path (path)
        - Requires: odd_text, file_path

        B) Code generator
        6) run_pipeline(json_path, user_requirements, output_path)
        - Purpose: generate/debug MESA code from conceptual model json.
        - Produces: model_code_path (path or directory)
        - Requires: conceptual_model_path (json_path), output_path
        - Optional: user_requirements

        C) Validation module
        7) evaluation_suggestion(conceptual_model)
        - Purpose: suggest VVUQ evaluation strategies.
        - Produces: evaluation_suggestions (json)
        - Requires: conceptual_model (or decision_rules/mechanistic_model)

        8) evaluation_code_generator(model_code, evaluation_suggestions, model_interface, output_path)
        - Purpose: generate evaluation code aligned with the model.
        - Produces: evaluation_code_path (path)
        - Requires: model_code, evaluation_suggestions, output_path
        - Optional: model_interface
        """



class RouterAgent:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model_name = model_name
        self.workspace: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []

        # ── Instantiate the sub-agents once ──
        self.model_constructor = ModelConstructor(model_name=model_name)
        self.coding_agent      = CodingAgent(model_name=model_name)
        self.model_validation  = ModelValidation(model_name=model_name)

    def _get_function_registry(self) -> Dict[str, Any]:
        return {
            "decision_rule_variables":   self.model_constructor.decision_rule_variables,
            "decision_rule_designer":    self.model_constructor.decision_rule_designer,
            "mechanism_translation":     self.model_constructor.mechanism_translation,
            "ODD_formatter":             self.model_constructor.ODD_formatter,
            "save_odd_to_wordfile":      self.model_constructor.save_odd_to_wordfile,
            "run_pipeline":              self.coding_agent.run_pipeline,
            "evaluation_suggestion":     self.model_validation.evaluation_suggestion,
            "evaluation_code_generator": self.model_validation.evaluation_code_generator,
        }

    def _call_function(self, name: str, args: str) -> str:
        """Execute a function and store its output in the workspace."""
        print(f"  → _call_function received: name={name}, args={args}")
        registry = self._get_function_registry()
        if name not in registry:
            return f"Error: function '{name}' not found."
        try:
            result = registry[name](**args)
            # Auto-save outputs to workspace by function name
            self.workspace[name + "_result"] = result
            # Also save under canonical keys for downstream functions
            OUTPUT_KEY_MAP = {
                "decision_rule_variables":   "variables",
                "decision_rule_designer":    "decision_rules",
                "mechanism_translation":     "mechanistic_model",
                "ODD_formatter":             "odd_text",
                "save_odd_to_wordfile":      "odd_docx_path",
                "run_pipeline":              "model_code_path",
                "evaluation_suggestion":     "evaluation_suggestions",
                "evaluation_code_generator": "evaluation_code_path",
            }
            if name in OUTPUT_KEY_MAP:
                self.workspace[OUTPUT_KEY_MAP[name]] = result
            return json.dumps(result) if not isinstance(result, str) else result
        except Exception as e:
            return f"Error running {name}: {e}"

    def chat(self, user_message: str) -> str:
        """Send a message and get a response, executing any tool calls along the way."""

        # Inject current workspace into the system prompt so LLM knows what's available
        system = ROUTER_SYSTEM_PROMPT + f"\n\nCurrent workspace:\n{json.dumps(self.workspace, indent=2, default=str)}"

        self.history.append({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": system}] + self.history

        # ── Agentic loop ──
        while True:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.0,
            )
            msg = response.choices[0].message
            print(f"  → stop_reason: {response.choices[0].finish_reason}")
            print(f"  → tool_calls: {msg.tool_calls}") 

            # No tool calls → final answer
            if not msg.tool_calls:
                self.history.append({"role": "assistant", "content": msg.content})
                return msg.content

            # Append assistant message (with tool calls) to history
            messages.append(msg)

            # Execute each tool call
            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                print(f"  ⚙ Calling {name}({list(args.keys())})")  # optional logging
                result = self._call_function(name, args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

if __name__ == "__main__":
    agent = RouterAgent(model_name="gpt-4o-mini")

    # Seed the workspace with your problem context upfront

    print("ABM Assistant ready. Type your request.\n")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        reply = agent.chat(user_input)
        print(f"\nAssistant: {reply}\n")