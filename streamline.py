import streamlit as st
from router import RouterAgent  # your file

st.title("XABM Interactive Planner")

# ── Persist the agent (and its workspace + history) across reruns ──
if "agent" not in st.session_state:
    st.session_state.agent = RouterAgent(model_name="gpt-4o-mini")

    # Seed workspace if you have a default problem context
    # st.session_state.agent.workspace["problem_context"] = "..."

agent = st.session_state.agent
st.session_state.agent.workspace["problem_context"] = "E:\LLM_for_abm\LLM-for-social-simualtion\model_construct_assistant\spiral_silence.txt"

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

# ── Handle new input ──
if user_input := st.chat_input("What would you like to do?"):
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = agent.chat(user_input)
        st.write(reply)