import streamlit as st
from router import RouterAgent  # your file
from pathlib import Path

st.title("XABM Interactive Planner")

BASE_DIR = Path(r"E:\LLM_for_abm\LLM-for-social-simualtion") # replace with your own path
MODEL_DIR    = BASE_DIR / "model_construct_assistant"  # don't change this
CODE_DIR     = BASE_DIR / "code_generator"
VAL_DIR      = BASE_DIR / "Validation_module"

# ── Persist the agent (and its workspace + history) across reruns ──
if "agent" not in st.session_state:
    st.session_state.agent = RouterAgent(model_name="gpt-4o-mini") # can replace with your own model
    ws = st.session_state.agent.workspace

    ### ----------------
    # Memory of the workspace, put all relevant input here
    # also state the path where you want to store your model, code, evaluation suggestions and evaluation code.
    # You can also give the link directly to LLM, but that may fail sometimes
    ### ----------------
    # Model construction
    ws["problem_context"]  = str(MODEL_DIR / "spiral_silence.txt")
    ws["model_save_path"]  = str(MODEL_DIR / "_spiral_silence_model2.json")
    ws["file_path"]        = str(MODEL_DIR / "spiral_silence_output.docx")

    # Code generator
    ws["user_requirements"] = str(CODE_DIR / "spiral_silence_requirement.txt")
    ws["code_output_path"]  = str(CODE_DIR / "code_test.py")
    ws["model_interface"]   = str(CODE_DIR / "code_test_interface.json")

    # Validator
    ws["evaluation_code_output_path"] = str(VAL_DIR / "spiral_silence_evaluation_code.py")

    greeting = st.session_state.agent.chat(
    "Introduce yourself and explain what functions are available, "
    "and tell the user what's already loaded in the workspace."
)
    st.session_state.greeting = greeting  # activate the greeting message at initialization

    # Seed workspace if you have a default problem context
    # st.session_state.agent.workspace["problem_context"] = "..."

agent = st.session_state.agent
ws = st.session_state.agent.workspace # reference only

with st.sidebar:
    st.subheader("Export Session")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("JSON"):
            path = agent.logger.export_json()
            st.success(f"Saved: {path}")

    with col2:
        if st.button("CSV"):
            path = agent.logger.export_csv()
            st.success(f"Saved: {path}")


if "greeting" in st.session_state:
    st.chat_message("assistant").write(st.session_state.greeting)

# ── Render conversation history ──
for msg in agent.history[2:]: # skip the initial greeting
    role = "user" if msg["role"] == "user" else "assistant"
    # Skip tool result messages — they're internal plumbing
    if msg["role"] in ("tool",):
        continue
    # Skip assistant messages that are just tool calls (no text content)
    content = msg.get("content")
    if not content:
        continue
    st.chat_message(role).write(content)

# ── Handle new input ──
if user_input := st.chat_input("What would you like to do?"):
    agent.logger.log("user", user_input) # log the messages of the user
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = agent.chat(user_input)
        st.write(reply)