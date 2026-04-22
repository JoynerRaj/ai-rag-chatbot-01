import streamlit as st
import requests
import json

FASTAPI_URL = "http://localhost:8000/audio"

st.set_page_config(page_title="Audio Event RAG", page_icon="🎧", layout="centered")

st.title("🎧 Audio Event RAG Chatbot")
st.markdown("Upload an audio file to extract acoustic events, and then ask questions about them!")

st.header("1. Upload Audio")
uploaded_file = st.file_uploader("Select an audio file (e.g., .mp3, .wav)", type=["mp3", "wav", "ogg", "m4a"])

if uploaded_file is not None:
    if st.button("Process Audio"):
        with st.spinner("Processing audio and extracting events..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "audio/mpeg")}
                res = requests.post(f"{FASTAPI_URL}/upload-audio/", files=files)
                if res.status_code == 200:
                    data = res.json()
                    st.success(f"✅ {data.get('message')}")
                else:
                    st.error(f"❌ Error: {res.text}")
            except Exception as e:
                st.error(f"Connection Error: Ensure FastAPI is running on port 8000. Details: {e}")

st.divider()

st.header("2. Ask Questions")
question = st.text_input("Ask a question (e.g., 'How many times did the dog bark?', 'What happened before 12 PM?')")

if question:
    if st.button("Ask Agent"):
        with st.spinner("Agent is reasoning..."):
            try:
                res = requests.post(f"{FASTAPI_URL}/ask/", json={"question": question})
                if res.status_code == 200:
                    data = res.json()
                    st.markdown(f"**🤖 Answer:**\n\n{data.get('answer')}")
                    
                    with st.expander("🔍 View Technical Details (SQL & Raw Data)"):
                        st.code(f"-- Generated SQL\n{data.get('executed_sql')}", language="sql")
                        st.json(data.get('raw_results'))
                else:
                    st.error(f"❌ Error: {res.text}")
            except Exception as e:
                st.error(f"Connection Error: Ensure FastAPI is running on port 8000. Details: {e}")
