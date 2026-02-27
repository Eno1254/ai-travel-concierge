import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.title("🌍 AI Travel Concierge (RAG Enabled)")

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    st.success("Document processed successfully!")

    query = st.text_input("Ask a question about the document:")

    if query:
        docs = vectorstore.similarity_search(query, k=3)

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
        Answer the question using ONLY the context below.

        Context:
        {context}

        Question:
        {query}
        """

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You answer based only on provided context."},
                {"role": "user", "content": prompt}
            ]
        )

        st.write(response.choices[0].message.content)