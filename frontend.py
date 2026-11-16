
import requests
import streamlit as st

st.title("🤖 Agentic RAG Assistant")

# ----- sidebar: add pages to the knowledge base -----
st.sidebar.header("Add URL to knowledge base")
url_text = st.sidebar.text_area("One URL per line")

if st.sidebar.button("Add to KB"):
    urls = url_text.split("\n")
    res = requests.post("http://127.0.0.1:8000/ingest", json={"urls": urls})
    st.sidebar.success(f"Added {res.json()['chunks_added']} chunks")

# ----- chat -----
question = st.chat_input("Ask a question...")

if question:
    st.chat_message("user").write(question)

    res = requests.post("http://127.0.0.1:8000/ask", json={"question": question})
    data = res.json()

    st.chat_message("assistant").write(data["answer"])
    st.caption("Source: " + data["source_used"])