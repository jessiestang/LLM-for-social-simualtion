# LLM-Enhanced Agent-Based Modelling: A Multi-Agent Framework for Transparent Social Simulation

## Introduction

Agent-based models (ABMs) are widely used to simulate complex social systems. They provide explict assumptions but limited flexibility. Recently, large language models (LLMs) have been explored as an alternative, as they enable more dynamic and human-like agent behavior. However, fully generative LLM-based simulations often lack transparency and control, due to the blackbox nature and intrinsic bias of LLMs.

This project proposed a LLM-enhanced framework that integrates structured agent-based modelling with LLM-driven components, aiming to provide a more **transparent and controllable alternative to generative ABMs**. We refer to it as **eXplainable agent-based model (XABM)**.

## Framework Overview

![LLM Framework](https://github.com/user-attachments/assets/36c7396f-bed3-411b-bbdd-6178f2d2a017)

The framework follows a **multi-agent architecture** in which LLMs are used as modular components rather than fully autonomous agents. A central orchestrating module coordinates interactions between different LLM-based agents and the human researcher (user).

The system consists of:

- **Orchestrator (central LLM/controller):** interprets simulation states and manages agent interactions
- **Task-specific LLM agents:** handle subtasks such as decision-making, reasoning, or communication
- **User (human researcher):** provide initial input; inspect and revise suggestions delivered by LLM agents

This modular design allows for better control over agent behavior while still leveraging the genertaive power of LLMs.

## Design Rationale

A key design decision was to **separate control and generation**:

- The orchestrator is responsible for coordination and decision flow
- Individual LLM agents handle specific reasoning tasks and provide explicit reasoning on why a decision has been made
- Human researchers have the final decisions on whether to adopt the work from the LLM agents or not

The result is a hybrid system that balances the interpretability of rule-based ABM and the generative power of LLMs.

## Implementation

The framework was implemented in Python using OpenAI-based APIs. Each agent was defined through targeted prompt engineering to ensure task-specific behavior, while the orchestrator dynamically selects and invokes agents through a tool-calling mechanism.

Key implementation aspects include:

- modular agent design
- prompt-based task specialization
- structured communication between components
- iterative debugging of agent coordination

## Challenges and Trade-offs

One of the main challenges was ensuring **consistent behavior across agents**. While modularity improves flexibility, it introduces coordination complexity.

Key trade-offs included:

- **flexibility vs control** in agent decision-making
- **modularity vs consistency** in outputs
- **interpretability vs generative power**

To address these, I refined prompt structures and constrained agent roles to maintain stable and predictable system behavior.

## Results and Insights

The framework demonstrates that LLM-enhanced ABMs can:

- improve behavioral flexibility compared to rule-based systems
- maintain better interpretability than fully generative approaches
- enable structured and explainable agent interactions

This approach provides a promising direction for combining **LLMs with traditional simulation frameworks**.

## Future Work

Future improvements include:

- integrate retrieval-augmented generation (RAG) for knowledge grounding
- add memory and planning mechanisms for long-term reasoning
- improve context management for scalability
- reduce the token cost induced by multi-agent communication

## Conclusion

This project shows how LLMs can be integrated into agent-based modelling in a controlled and modular way. By combining explanatory clarity with LLM-driven reasoning, the XABM framework offers a more transparent and flexible approach to modelling complex social systems.

## Repository Structure

```
LLM-for-social-simulation/
│
├── streamline.py               # Entry point — launches the Streamlit web UI and wires the RouterAgent to the chat interface
├── router.py                   # Central orchestrator (RouterAgent) — interprets user requests, dispatches tool calls to sub-agents, and maintains a shared workspace across the session
├── logger.py                   # SessionLogger — records every user message, tool call, and agent response; exports sessions to JSON or CSV
├── generated_model.py          # Example output — a Mesa ABM file produced by the framework (updated each time the code generator runs)
│
├── model_construct_assistant/  # Sub-agent that translates a plain-text problem description into a structured conceptual model (decision variables → rules → mechanistic model → ODD document)
├── code_generator/             # Sub-agent that generates and iteratively debugs Mesa/Python simulation code from the conceptual model JSON
├── Validation_module/          # Sub-agent that suggests VVUQ (Verification, Validation & Uncertainty Quantification) strategies and generates the corresponding evaluation code
│
├── logs/                       # Auto-generated session logs (JSON / CSV) exported by SessionLogger
├── output_plots/               # Plots produced during model runs and validation experiments
│
├── requirements.txt            # Python dependencies
└── LICENSE
```

### Root-level files

| File | Role |
|---|---|
| `streamline.py` | Streamlit front end. Starts the chat UI, seeds the agent workspace with input/output file paths, and exposes session-export buttons (JSON / CSV). This is the only file users need to run directly. |
| `router.py` | The brain of the framework. `RouterAgent` receives each user message, asks the LLM to decide which tool to call, executes the matching sub-agent function, stores results in a shared workspace, and loops until a final answer is ready. |
| `logger.py` | `SessionLogger` timestamps and records every event (user turns, tool calls, results). Call `export_json()` or `export_csv()` to dump the full session to the `logs/` folder. |
| `generated_model.py` | A ready-to-run Mesa ABM file written by the code generator. Treat it as output, not source — it is overwritten each time the pipeline produces a new model. |

### Folders

| Folder | Contents |
|---|---|
| `model_construct_assistant/` | Code and prompts for the model-construction sub-agent: variable extraction, decision-rule design, mechanistic model translation, and ODD-protocol formatting. |
| `code_generator/` | Code and prompts for the code-generation sub-agent: Mesa code synthesis, automated debugging loop, and example generated models. |
| `Validation_module/` | Code and prompts for the validation sub-agent: VVUQ strategy suggestion and evaluation-code generation. |
| `logs/` | Session log files (one JSON and/or CSV per run), written automatically by `SessionLogger`. |
| `output_plots/` | Figures saved during model execution and evaluation (e.g., convergence plots, sensitivity analysis charts). |

## How to Run the Framework

Install all the required dependencies by running the following code:

```
pip install -r requirements.txt
```

To be able to run the framework, an OpenAI API is needed. Run the following code to store it as an environment variable:

```
$env:OPENAI_API_KEY = "YOUR API KEY"
[System.Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "YOUR API KEY HERE", "User")
```

Make sure all required input files (e.g., problem formulation, user requirement for coding) and directories for output saving are stored properly in the working space in the streamline.py file. Run the following line in your terminal to run the front end:

```
streamlit run streamline.py
```
