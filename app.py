import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

from tools.web_search import web_search
from tools.disease_info import disease_info

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="AI Medical Assistant")

st.title("🩺 AI Medical Assistant")
st.warning("This AI is for educational purposes only. Consult a doctor for medical advice.")

# -------------------------
# Sidebar
# -------------------------

st.sidebar.title("Medical Tools")

uploaded_file = st.sidebar.file_uploader(
    "Upload Medical Report (PDF)",
    type="pdf"
)

st.sidebar.markdown("### Example Questions")

st.sidebar.write("""
• What causes diabetes?  
• Symptoms of dengue  
• Latest malaria treatment  
• Explain my blood test report
""")

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# -------------------------
# Session State
# -------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# -------------------------
# Process PDF
# -------------------------

if uploaded_file:

    with open("report.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader("report.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    documents = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings()

    st.session_state.vectorstore = FAISS.from_documents(documents, embeddings)

    st.success("Medical report processed successfully")

# -------------------------
# Show chat history
# -------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# Chat input
# -------------------------

query = st.chat_input("Ask a medical question...")

if query:

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    context = ""
    tool_result = ""

    # -------------------------
    # RAG Tool
    # -------------------------

    if st.session_state.vectorstore:

        docs = st.session_state.vectorstore.similarity_search(query, k=3)

        context = "\n".join([doc.page_content for doc in docs])

    # -------------------------
    # Web Search Tool
    # -------------------------

    if "latest" in query or "news" in query or "research" in query:
        tool_result = web_search(query)

    # -------------------------
    # Disease Info Tool
    # -------------------------

    elif "dengue" in query or "malaria" in query or "diabetes" in query:
        tool_result = disease_info(query)

    # -------------------------
    # Prompt
    # -------------------------

    prompt = f"""
You are a helpful medical assistant.

Use available information to answer.

Tool Result:
{tool_result}

Medical Report Context:
{context}

User Question:
{query}

Rules:
- Explain in simple language
- Do not give medical diagnosis
- Suggest consulting a doctor if needed
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)