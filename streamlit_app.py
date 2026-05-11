import streamlit as st
import requests

API_URL = "http://localhost:8000"

# ── Page Config ──────────────────────────────────
st.set_page_config(
    page_title="Healthcare Knowledge Assistant",
    page_icon="🏥",
    layout="wide"
)

# ── Header ───────────────────────────────────────
st.title("🏥 Healthcare Knowledge Assistant")
st.markdown("*Powered by RAG + LangGraph Multi-Agents + Gemini*")
st.divider()

# ── Mode Selection ───────────────────────────────

mode = "Multi-Agent RAG"
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
                    - **Retriever Agent** — Fetches medical info
                    - **Consultation Agent** — Generates response
                    - **Diagnosis Agent** — Suggests conditions
                """)

# ── Chat History ─────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat Input ───────────────────────────────────
if question := st.chat_input("Ask a medical question..."):
    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing your question..."):
            try:
                response = requests.post(
                    f"{API_URL}/agent-query",
                    json={"question": question}
                )
                data = response.json()
                # Show agent outputs
                with st.expander(
                     "🔍 Retriever Agent Output"
                ):
                    st.write(data["retrieved_context"])
                    with st.expander(

                        "👨‍⚕️ Consultation Agent Output"

                    ):

                        st.write(data["consultation_response"])

                    with st.expander(

                        "🩺 Diagnosis Agent Output"

                    ):

                        st.write(data["diagnosis_result"])

                    st.markdown("### 📋 Final Response")

                    st.markdown(data["final_response"])

                    answer = data["final_response"]

                st.session_state.messages.append({

                    "role": "assistant",

                    "content": answer

                })

            except Exception as e:

                st.error(f"Error: {str(e)}")
st.info("Make sure FastAPI is running!")

# ── Footer ───────────────────────────────────────

st.divider()

st.caption(

    "⚠️ For educational purposes only. "

    "Always consult a qualified healthcare professional."

)
 