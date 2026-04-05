import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
import logging

from utils.db import *

# ---------------- LOGGING ----------------
logging.basicConfig(
    filename="app.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------- STYLE ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b0f19, #111827);
}
.glass {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 15px;
    padding: 20px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
}
.stButton>button {
    border-radius: 10px;
    background: linear-gradient(135deg, #6d28d9, #9333ea);
    color: white;
}
section[data-testid="stSidebar"] {
    background: #111827;
}
</style>
""", unsafe_allow_html=True)

# ---------------- API ----------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("API KEY missing (.env)")
    st.stop()

client = Groq(api_key=api_key)

# ---------------- DB ----------------
create_tables()

# ---------------- SESSION ----------------
if "user" not in st.session_state:
    st.session_state.user = None

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

if "last_question" not in st.session_state:
    st.session_state.last_question = ""

# ---------------- LOGIN ----------------
if st.session_state.user is None:
    st.title("🔐 Login / Signup")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns(2)

    if col1.button("Login"):
        if not username or not password:
            st.warning("Fill all fields")
        else:
            user = login(username, password)
            if user:
                st.session_state.user = user
                st.rerun()
            else:
                st.error("Invalid login")

    if col2.button("Signup"):
        if not username or not password:
            st.warning("Fill all fields")
        else:
            if signup(username, password):
                st.success("Account created")
            else:
                st.error("User exists")

# ---------------- MAIN ----------------
else:
    user_id, role = st.session_state.user

    # Sidebar
    st.sidebar.title("User Panel")

    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

    if st.sidebar.button("🧹 Clear Chat"):
        st.session_state.last_answer = ""
        st.session_state.last_question = ""
        st.session_state.selected_q = ""
        st.session_state.selected_a = ""
        st.rerun()

    if st.sidebar.button("🗑 Clear My Data"):
        clear_user_history(user_id)
        st.success("Data cleared")

    # File upload validation
    st.sidebar.subheader("📄 Upload Report")
    file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

    if file:
        if file.type != "application/pdf":
            st.sidebar.error("Only PDF supported")
        else:
            st.sidebar.success("PDF uploaded")

    # Filter
    st.sidebar.subheader("🔍 Filter")
    search = st.sidebar.text_input("Search history")

    # History
    st.sidebar.subheader("📜 History")
    history = get_user_history(user_id)

    for i, (q, a) in enumerate(history[::-1]):
        if search.lower() in q.lower():
            if st.sidebar.button(q[:25] + "...", key=f"hist_{i}"):
                st.session_state.selected_q = q
                st.session_state.selected_a = a

    # Main UI
    st.title("🩺 AI Medical Assistant")

    question = st.text_input("Ask your question")

    # Input validation
    if question:
        if not question.strip():
            st.warning("Enter valid question")
            st.stop()

        st.session_state.last_question = question

        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": """
You are a medical assistant.

Answer in structured format:
1. Definition
2. Causes
3. Symptoms
4. Prevention
5. When to see a doctor
"""
                    },
                    {"role": "user", "content": question}
                ]
            )

            answer = response.choices[0].message.content
            st.session_state.last_answer = answer

            save_history(user_id, question, answer)

        except Exception as e:
            logging.error(str(e))
            st.error("AI Error occurred")

    # Show answer
    if st.session_state.last_answer:
        st.markdown(f"""
        <div class="glass">
        <h4>🧠 AI Answer</h4>
        <p>{st.session_state.last_answer}</p>
        </div>
        """, unsafe_allow_html=True)

    # Selected history
    if "selected_q" in st.session_state and st.session_state.selected_q:
        st.markdown("### 📌 Selected History")
        st.write("Q:", st.session_state.selected_q)
        st.write("A:", st.session_state.selected_a)

    # Admin panel
    if role == "admin":
        st.markdown("## 🔐 Admin Dashboard")

        search_user = st.text_input("Search user")

        data = get_all_history()

        for u, q, a in data:
            if search_user.lower() in u.lower():
                st.markdown(f"""
                <div class="glass">
                <b>👤 {u}</b><br>
                Q: {q}<br>
                A: {a}
                </div>
                """, unsafe_allow_html=True)