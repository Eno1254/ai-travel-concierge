from tools.web_search import web_search
from tools.disease_info import disease_info
import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# -----------------------
# Setup
# -----------------------

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="AI Medical Assistant")

st.title("🩺 AI Medical Assistant")
st.warning("⚠ This AI provides educational information only. Always consult a doctor.")

# -----------------------
# Sidebar
# -----------------------

st.sidebar.title("🩺 Medical Tools")

st.sidebar.write("Upload a medical report or ask a health question.")

uploaded_file = st.sidebar.file_uploader(
    "📄 Upload Medical Report (PDF)",
    type="pdf"
)

st.sidebar.markdown("### 💡 Example Questions")

st.sidebar.write("""
• What causes high blood pressure?  
• What are symptoms of diabetes?  
• Explain my blood test results  
• Is high cholesterol dangerous?  
""")

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# -----------------------
# Session state
# -----------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------
# Process uploaded PDF
# -----------------------

if uploaded_file:

    with open("report.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader("report.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()

    st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)

    st.success("Medical report processed successfully!")

# -----------------------
# Show chat history
# -----------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------
# Chat input
# -----------------------

query = st.chat_input("Ask a medical question...")

if query:

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    # -----------------------
    # If report exists → use RAG
    # -----------------------

    if st.session_state.vectorstore:

        docs = st.session_state.vectorstore.similarity_search(query, k=3)

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are a medical report assistant.

Explain medical information in simple language.

Rules:
- Use the medical report context if relevant
- Explain abnormal values if found
- Do NOT give diagnosis
- Suggest consulting a doctor if needed

Medical Report Context:
{context}

Question:
{query}
"""

    # -----------------------
    # Otherwise general medical AI
    # -----------------------

    else:

        prompt = f"""
You are a helpful medical assistant.

Answer general health questions clearly.

Rules:
- Provide educational information
- Do NOT give medical diagnosis
- Suggest consulting a doctor for serious concerns

Question:
{query}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):
        st.markdown(answer)