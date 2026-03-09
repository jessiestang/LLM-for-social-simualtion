import streamlit as st
from router import RouterAgent  # your file

st.title("XABM Interactive Planner")

# ── Persist the agent (and its workspace + history) across reruns ──
if "agent" not in st.session_state:
    st.session_state.agent = RouterAgent(model_name="gpt-4o-mini")

    # Seed workspace if you have a default problem context
    # st.session_state.agent.workspace["problem_context"] = "..."

agent = st.session_state.agent

### ----------------
# Memory of the workspace, put all relevant input here
# also state the path where you want to store your model, code, evaluation suggestions and evaluation code.
# You can also give the link directly to LLM, but that may fail sometimes
### ----------------
# required to run the model construction assistant
st.session_state.agent.workspace["problem_context"] = "E:\LLM_for_abm\LLM-for-social-simualtion\model_construct_assistant\spiral_silence.txt"
st.session_state.agent.workspace["model_save_path"] = "E:\LLM_for_abm\LLM-for-social-simualtion\model_construct_assistant\_spiral_silence_model2.json"
st.session_state.agent.workspace["file_path"] = "E:\LLM_for_abm\LLM-for-social-simualtion\model_construct_assistant\spiral_silence_output.docx"

# required to run the code generator
st.session_state.agent.workspace["user_requirements"] = "E:\LLM_for_abm\LLM-for-social-simualtion\code_generator\spiral_silence_requirement.txt"
st.session_state.agent.workspace["code_output_path"] = "E:\LLM_for_abm\LLM-for-social-simualtion\code_generator\code_test.py"
st.session_state.agent.workspace["model_interface"] = "E:\LLM_for_abm\LLM-for-social-simualtion\code_generator\spiral_silence_interface.json"

# required to run the validator
st.session_state.agent.workspace["evaluation_code_output_path"] = "E:\LLM_for_abm\LLM-for-social-simualtion\Validation_module\spiral_silence_evaluation_code.py"
 
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

# ── Render conversation history ──
for msg in agent.history:
    role = "user" if msg["role"] == "user" else "assistant"
    # Skip tool result messages — they're internal plumbing
    if msg["role"] in ("tool",):
        continue
    # Skip assistant messages that are just tool calls (no text content)
    content = msg.get("content")
    if not content:
        continue
    st.chat_message(role).write(content)

# Trigger greeting once on first load

greeting = st.session_state.agent.chat(
    "Introduce yourself and explain what functions are available, "
    "and tell the user what's already loaded in the workspace."
)
st.session_state.greeting = greeting

if "greeting" in st.session_state:
    st.chat_message("assistant").write(st.session_state.greeting)
# ── Handle new input ──
if user_input := st.chat_input("What would you like to do?"):
    agent.logger.log("user", user_input) # log the messages of the user
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = agent.chat(user_input)
        st.write(reply)