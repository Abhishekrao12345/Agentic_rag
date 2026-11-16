import time
import threading

import requests
import streamlit as st
import uvicorn
from dotenv import load_dotenv

load_dotenv(encoding="utf-8-sig")


# ----- start the FastAPI server inside this app (runs once) -----
@st.cache_resource
def start_api():
    from main import api    # this loads backend.py: model + ingestion

    server = uvicorn.Server(
        uvicorn.Config(api, host="127.0.0.1", port=8000, log_level="warning")
    )
    threading.Thread(target=server.run, daemon=True).start()

    # wait until the API answers
    for _ in range(60):
        try:
            requests.get("http://127.0.0.1:8000/docs", timeout=1)
            return True
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    return False


with st.spinner("Starting the agent..."):
    api_ready = start_api()

if not api_ready:
    st.error("The API did not start. Check the app logs.")
    st.stop()


# ----- page -----
st.title("🤖 Agentic RAG Assistant")

# ----- sidebar: add pages to the knowledge base -----
st.sidebar.header("Add URL to knowledge base")
url_text = st.sidebar.text_area("One URL per line")

if st.sidebar.button("Add to KB"):
    urls = [u.strip() for u in url_text.split("\n") if u.strip()]
    res = requests.post("http://127.0.0.1:8000/ingest", json={"urls": urls})
    st.sidebar.success(f"Added {res.json()['chunks_added']} chunks")

# ----- chat -----
question = st.chat_input("Ask a question...")

if question:
    st.chat_message("user").write(question)

    with st.spinner("Thinking..."):
        res = requests.post("http://127.0.0.1:8000/ask", json={"question": question})
        data = res.json()

    st.chat_message("assistant").write(data["answer"])
    st.caption("Source: " + data["source_used"])