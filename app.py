import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="AI Travel Concierge")
st.title("🌍 AI Travel Concierge (RAG Enabled)")

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found in environment variables.")
    st.stop()

client = Groq(api_key=groq_api_key)

# -----------------------------
# SESSION STATE
# -----------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()

    st.session_state.vectorstore = FAISS.from_documents(texts, embeddings)

    st.success("PDF processed successfully!")

# -----------------------------
# CHAT DISPLAY
# -----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# CHAT INPUT
# -----------------------------
if st.session_state.vectorstore:

    query = st.chat_input("Ask a question about the uploaded PDF")

    if query:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Retrieve context
        docs = st.session_state.vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are an AI assistant.

Use the context below to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{query}
"""

        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )

            answer = response.choices[0].message.content

        # Show assistant message
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.markdown(answer)

else:
    st.info("Upload a PDF to start chatting.")
