import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
import logging
import hashlib
import time
import traceback # Added for better error logging

from utils.db import *

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="MediAI Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOGGING ----------------
logging.basicConfig(
    filename="app.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------- STYLE ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

/* ===== RESET & BASE ===== */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    font-family: 'DM Sans', sans-serif;
    background: #05070f;
    color: #e8eaf0;
}

/* ===== ANIMATED BACKGROUND ===== */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(0, 200, 180, 0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 80% at 80% 90%, rgba(99, 102, 241, 0.08) 0%, transparent 60%),
        radial-gradient(ellipse 40% 40% at 60% 30%, rgba(16, 185, 129, 0.04) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: #080c18 !important;
    border-right: 1px solid rgba(0, 200, 180, 0.12) !important;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

/* ===== HIDE STREAMLIT CHROME ===== */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* ===== TYPOGRAPHY ===== */
h1, h2, h3, h4 {
    font-family: 'Syne', sans-serif;
    letter-spacing: -0.02em;
}

/* ===== LOGIN PAGE ===== */
.login-wrapper {
    min-height: 85vh;
    display: flex;
    align-items: center;
    justify-content: center;
}

.login-card {
    width: 100%;
    max-width: 440px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(0, 200, 180, 0.15);
    border-radius: 24px;
    padding: 48px 44px 44px;
    backdrop-filter: blur(20px);
    box-shadow:
        0 0 0 1px rgba(0,200,180,0.05),
        0 24px 80px rgba(0,0,0,0.6),
        inset 0 1px 0 rgba(255,255,255,0.05);
    animation: cardIn 0.6s cubic-bezier(0.22, 1, 0.36, 1) both;
}

@keyframes cardIn {
    from { opacity: 0; transform: translateY(32px) scale(0.97); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}

.login-logo {
    width: 56px;
    height: 56px;
    background: linear-gradient(135deg, #00c8b4, #10b981);
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    margin: 0 auto 20px;
    box-shadow: 0 8px 32px rgba(0, 200, 180, 0.35);
}

.login-title {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 800;
    text-align: center;
    color: #f0f4ff;
    margin-bottom: 6px;
    letter-spacing: -0.03em;
}

.login-sub {
    font-size: 14px;
    color: rgba(200, 210, 230, 0.5);
    text-align: center;
    margin-bottom: 36px;
    font-weight: 300;
}

.tab-switcher {
    display: flex;
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 4px;
    margin-bottom: 28px;
    border: 1px solid rgba(255,255,255,0.06);
}

.tab-btn {
    flex: 1;
    padding: 9px 0;
    border: none;
    border-radius: 9px;
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.25s ease;
}

.tab-btn.active {
    background: linear-gradient(135deg, #00c8b4, #10b981);
    color: #fff;
    box-shadow: 0 4px 16px rgba(0, 200, 180, 0.3);
}

.tab-btn.inactive {
    background: transparent;
    color: rgba(200, 220, 230, 0.45);
}

/* ===== INPUTS ===== */
.stTextInput > div > div > input {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.09) !important;
    border-radius: 12px !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    padding: 12px 16px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

.stTextInput > div > div > input:focus {
    border-color: rgba(0, 200, 180, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(0, 200, 180, 0.1) !important;
    outline: none !important;
}

.stTextInput > div > div > input::placeholder {
    color: rgba(200, 220, 230, 0.28) !important;
}

.stTextInput label {
    color: rgba(200, 220, 240, 0.6) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ===== BUTTONS ===== */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #00c8b4, #10b981) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    padding: 13px 24px !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(0, 200, 180, 0.25) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(0, 200, 180, 0.4) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* ===== SIDEBAR BUTTONS ===== */
section[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.05) !important;
    color: rgba(200,220,240,0.7) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    box-shadow: none !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 10px 16px !important;
    text-align: left !important;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(0, 200, 180, 0.1) !important;
    border-color: rgba(0, 200, 180, 0.25) !important;
    color: #00c8b4 !important;
    transform: none !important;
}

/* ===== SIDEBAR BRAND ===== */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0 4px 24px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 20px;
}

.sidebar-logo {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, #00c8b4, #10b981);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
    box-shadow: 0 4px 16px rgba(0,200,180,0.3);
}

.sidebar-brand-text {
    font-family: 'Syne', sans-serif;
    font-size: 16px;
    font-weight: 700;
    color: #f0f4ff;
    letter-spacing: -0.02em;
}

.sidebar-brand-sub {
    font-size: 11px;
    color: rgba(200,220,230,0.4);
    font-weight: 300;
}

/* ===== USER BADGE ===== */
.user-badge {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 14px;
    background: rgba(0, 200, 180, 0.07);
    border: 1px solid rgba(0, 200, 180, 0.15);
    border-radius: 12px;
    margin-bottom: 20px;
}

.user-avatar {
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, #00c8b4, #10b981);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    font-weight: 700;
    color: #fff;
    font-family: 'Syne', sans-serif;
    flex-shrink: 0;
}

.user-info-name {
    font-size: 13px;
    font-weight: 600;
    color: #e8eaf0;
    font-family: 'Syne', sans-serif;
}

.user-info-role {
    font-size: 11px;
    color: rgba(200,220,230,0.45);
}

/* ===== SECTION LABELS ===== */
.sidebar-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(200,220,240,0.3);
    padding: 0 4px;
    margin: 16px 0 8px;
}

/* ===== HISTORY ITEMS ===== */
.hist-item {
    padding: 10px 12px;
    border-radius: 10px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 6px;
    cursor: pointer;
    transition: all 0.2s ease;
    font-size: 12px;
    color: rgba(200,220,240,0.55);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.hist-item:hover {
    background: rgba(0,200,180,0.08);
    border-color: rgba(0,200,180,0.2);
    color: #00c8b4;
}

/* ===== MAIN CHAT AREA ===== */
.page-header {
    margin-bottom: 32px;
}

.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 32px;
    font-weight: 800;
    color: #f0f4ff;
    letter-spacing: -0.03em;
    line-height: 1.1;
}

.page-title span {
    background: linear-gradient(135deg, #00c8b4, #10b981);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.page-subtitle {
    font-size: 15px;
    color: rgba(200, 220, 240, 0.45);
    margin-top: 6px;
    font-weight: 300;
}

/* ===== QUESTION INPUT AREA ===== */
.input-container {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 28px;
    backdrop-filter: blur(10px);
}

/* ===== ANSWER CARD ===== */
.answer-card {
    background: rgba(0, 200, 180, 0.04);
    border: 1px solid rgba(0, 200, 180, 0.15);
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 20px;
    animation: fadeUp 0.5s cubic-bezier(0.22, 1, 0.36, 1) both;
    position: relative;
    overflow: hidden;
}

.answer-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00c8b4, #10b981, transparent);
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

.answer-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
    padding-bottom: 16px;
    border-bottom: 1px solid rgba(0, 200, 180, 0.1);
}

.answer-header-icon {
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, #00c8b4, #10b981);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 15px;
}

.answer-header-title {
    font-family: 'Syne', sans-serif;
    font-size: 14px;
    font-weight: 700;
    color: #00c8b4;
    letter-spacing: 0.02em;
}

.answer-header-q {
    font-size: 12px;
    color: rgba(200,220,240,0.4);
    margin-left: auto;
    max-width: 300px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.answer-body {
    font-size: 14px;
    line-height: 1.75;
    color: rgba(220, 235, 245, 0.82);
    white-space: pre-wrap;
}

/* ===== QUICK EXAMPLES ===== */
.examples-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    margin-top: 16px;
}

.example-chip {
    padding: 12px 16px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    font-size: 13px;
    color: rgba(200,220,240,0.55);
    cursor: pointer;
    transition: all 0.2s ease;
    font-family: 'DM Sans', sans-serif;
}

.example-chip:hover {
    background: rgba(0,200,180,0.08);
    border-color: rgba(0,200,180,0.25);
    color: #00c8b4;
    transform: translateY(-1px);
}

/* ===== STATS STRIP ===== */
.stats-strip {
    display: flex;
    gap: 12px;
    margin-bottom: 28px;
}

.stat-card {
    flex: 1;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 16px 20px;
    text-align: center;
}

.stat-number {
    font-family: 'Syne', sans-serif;
    font-size: 24px;
    font-weight: 800;
    color: #00c8b4;
}

.stat-label {
    font-size: 11px;
    color: rgba(200,220,240,0.4);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 2px;
}

/* ===== ADMIN ===== */
.admin-banner {
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.1));
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 14px;
}

.admin-banner-icon {
    font-size: 28px;
}

.admin-banner-title {
    font-family: 'Syne', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: #a5b4fc;
}

.admin-banner-sub {
    font-size: 13px;
    color: rgba(165,180,252,0.6);
}

.admin-row {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 10px;
    transition: border-color 0.2s ease;
}

.admin-row:hover {
    border-color: rgba(99,102,241,0.2);
}

.admin-row-user {
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 700;
    color: #a5b4fc;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.admin-row-q {
    font-size: 13px;
    color: rgba(220,235,245,0.7);
    margin-bottom: 6px;
    padding-left: 4px;
}

.admin-row-a {
    font-size: 12px;
    color: rgba(200,220,240,0.45);
    padding-left: 4px;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

/* ===== SELECTED HISTORY ===== */
.selected-hist-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px;
    padding: 22px 26px;
    margin-bottom: 20px;
}

.selected-hist-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(200,220,240,0.3);
    margin-bottom: 12px;
}

.selected-hist-q {
    font-size: 15px;
    font-weight: 600;
    color: #e8eaf0;
    font-family: 'Syne', sans-serif;
    margin-bottom: 12px;
}

.selected-hist-a {
    font-size: 13px;
    color: rgba(200,220,240,0.6);
    line-height: 1.65;
    white-space: pre-wrap;
}

/* ===== ALERTS / INFO ===== */
.stAlert {
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
}

/* ===== FILE UPLOADER ===== */
.stFileUploader > div {
    border-radius: 12px !important;
    border-color: rgba(255,255,255,0.1) !important;
    background: rgba(255,255,255,0.02) !important;
}

/* ===== SEARCH INPUT ===== */
.search-wrap .stTextInput > div > div > input {
    padding-left: 36px !important;
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,200,180,0.2); border-radius: 4px; }

/* ===== DIVIDER ===== */
.sidebar-divider {
    height: 1px;
    background: rgba(255,255,255,0.06);
    margin: 16px 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HELPERS ----------------
# Kept as SHA256 to ensure compatibility with your existing db.py logic. 
# Switch to bcrypt in the future for production-grade security!
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Callback for Quick Examples
def set_example(ex_text):
    st.session_state.question_input = ex_text
    st.session_state.example_trigger = True

# ---------------- API ----------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("⚠️ GROQ_API_KEY missing from .env file")
    st.stop()

client = Groq(api_key=api_key)

# ---------------- DB ----------------
create_tables()

# ---------------- SESSION ----------------
defaults = {
    "user": None,
    "username": "",
    "last_answer": "",
    "last_question": "",
    "selected_q": "",
    "selected_a": "",
    "login_tab": "login",
    "loading": False,
    "example_trigger": False # Added for callback logic
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ================================================================
# LOGIN PAGE
# ================================================================
if st.session_state.user is None:

    col_l, col_c, col_r = st.columns([1, 1.1, 1])

    with col_c:
        st.markdown("<div style='height:48px'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:center; margin-bottom:36px;">
            <div style="
                width:64px; height:64px;
                background: linear-gradient(135deg,#00c8b4,#10b981);
                border-radius:18px;
                display:flex; align-items:center; justify-content:center;
                font-size:32px; margin:0 auto 16px;
                box-shadow:0 8px 40px rgba(0,200,180,0.4);
            ">🩺</div>
            <div style="
                font-family:'Syne',sans-serif;
                font-size:30px; font-weight:800;
                color:#f0f4ff; letter-spacing:-0.03em;
                margin-bottom:6px;
            ">MediAI</div>
            <div style="font-size:14px;color:rgba(200,220,240,0.4);font-weight:300;">
                Your intelligent medical assistant
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Tab switcher
        tab_col1, tab_col2 = st.columns(2)
        with tab_col1:
            if st.button("Sign In", key="tab_login", use_container_width=True):
                st.session_state.login_tab = "login"
        with tab_col2:
            if st.button("Create Account", key="tab_signup", use_container_width=True):
                st.session_state.login_tab = "signup"

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Card container
        with st.container():
            if st.session_state.login_tab == "login":
                st.markdown("""
                <div style="
                    background:rgba(255,255,255,0.03);
                    border:1px solid rgba(0,200,180,0.14);
                    border-radius:20px; padding:32px 28px;
                    box-shadow:0 24px 80px rgba(0,0,0,0.5);
                ">
                    <div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;color:#f0f4ff;margin-bottom:4px;">Welcome back</div>
                    <div style="font-size:13px;color:rgba(200,220,240,0.4);margin-bottom:24px;">Sign in to continue your health journey</div>
                </div>
                """, unsafe_allow_html=True)

                username = st.text_input("Username", placeholder="Enter your username", key="li_user")
                password = st.text_input("Password", type="password", placeholder="Enter your password", key="li_pass")

                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

                if st.button("Sign In →", key="btn_login", use_container_width=True):
                    if not username.strip() or not password.strip():
                        st.warning("Please fill in all fields.")
                    else:
                        hashed = hash_password(password)
                        user = login(username.strip(), hashed)
                        if user:
                            st.session_state.user = user
                            st.session_state.username = username.strip()
                            st.rerun()
                        else:
                            # Try legacy unhashed (fallback)
                            user = login(username.strip(), password)
                            if user:
                                st.session_state.user = user
                                st.session_state.username = username.strip()
                                st.rerun()
                            else:
                                st.error("Invalid username or password.")

            else:
                st.markdown("""
                <div style="
                    background:rgba(255,255,255,0.03);
                    border:1px solid rgba(0,200,180,0.14);
                    border-radius:20px; padding:32px 28px;
                    box-shadow:0 24px 80px rgba(0,0,0,0.5);
                ">
                    <div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;color:#f0f4ff;margin-bottom:4px;">Create account</div>
                    <div style="font-size:13px;color:rgba(200,220,240,0.4);margin-bottom:24px;">Join MediAI for personalized health assistance</div>
                </div>
                """, unsafe_allow_html=True)

                new_user = st.text_input("Username", placeholder="Choose a username", key="su_user")
                new_pass = st.text_input("Password", type="password", placeholder="Choose a password (min 6 chars)", key="su_pass")
                new_pass2 = st.text_input("Confirm Password", type="password", placeholder="Repeat your password", key="su_pass2")

                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

                if st.button("Create Account →", key="btn_signup", use_container_width=True):
                    if not new_user.strip() or not new_pass.strip():
                        st.warning("Please fill in all fields.")
                    elif len(new_pass) < 6:
                        st.warning("Password must be at least 6 characters.")
                    elif new_pass != new_pass2:
                        st.error("Passwords do not match.")
                    else:
                        hashed = hash_password(new_pass)
                        if signup(new_user.strip(), hashed):
                            st.success("✅ Account created! Please sign in.")
                            st.session_state.login_tab = "login"
                            time.sleep(0.8)
                            st.rerun()
                        else:
                            st.error("Username already taken. Try another.")

        st.markdown("""
        <div style="text-align:center;margin-top:24px;font-size:12px;color:rgba(200,220,240,0.2);">
            🔒 Your data is private and encrypted
        </div>
        """, unsafe_allow_html=True)

# ================================================================
# MAIN APP
# ================================================================
else:
    user_id, role = st.session_state.user
    uname = st.session_state.get("username", "User")
    history = get_user_history(user_id)

    # ===== SIDEBAR =====
    with st.sidebar:
        # Brand
        st.markdown(f"""
        <div class="sidebar-brand">
            <div class="sidebar-logo">🩺</div>
            <div>
                <div class="sidebar-brand-text">MediAI</div>
                <div class="sidebar-brand-sub">Medical Assistant</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # User badge
        initial = uname[0].upper() if uname else "U"
        role_label = "Administrator" if role == "admin" else "Member"
        st.markdown(f"""
        <div class="user-badge">
            <div class="user-avatar">{initial}</div>
            <div>
                <div class="user-info-name">{uname}</div>
                <div class="user-info-role">{role_label}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Actions
        st.markdown('<div class="sidebar-label">Actions</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 Clear Chat", use_container_width=True):
                st.session_state.last_answer = ""
                st.session_state.last_question = ""
                st.session_state.selected_q = ""
                st.session_state.selected_a = ""
                st.rerun()
        with col2:
            if st.button("🗑 My Data", use_container_width=True):
                clear_user_history(user_id)
                st.session_state.last_answer = ""
                st.session_state.last_question = ""
                st.rerun()

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # PDF Upload
        st.markdown('<div class="sidebar-label">Upload Report</div>', unsafe_allow_html=True)
        file = st.file_uploader("PDF only", type=["pdf"], label_visibility="collapsed")
        if file:
            if file.type != "application/pdf":
                st.error("Only PDF supported")
            else:
                st.success(f"📄 {file.name[:22]}...")

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # History
        st.markdown('<div class="sidebar-label">Search History</div>', unsafe_allow_html=True)
        search = st.text_input("Search", placeholder="🔍  Search questions...", label_visibility="collapsed", key="search_hist")

        st.markdown('<div class="sidebar-label">Recent</div>', unsafe_allow_html=True)

        if not history:
            st.markdown('<div style="font-size:12px;color:rgba(200,220,240,0.25);padding:8px 4px;">No history yet</div>', unsafe_allow_html=True)
        else:
            for i, (q, a) in enumerate(history[::-1][:15]):
                if search.lower() in q.lower():
                    label = q[:30] + "…" if len(q) > 30 else q
                    if st.button(f"💬 {label}", key=f"hist_{i}", use_container_width=True):
                        st.session_state.selected_q = q
                        st.session_state.selected_a = a
                        st.session_state.last_question = q
                        st.session_state.last_answer = a
                        st.rerun()

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        if st.button("⎋  Sign Out", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    # ===== MAIN CONTENT =====
    st.markdown(f"""
    <div class="page-header">
        <div class="page-title">AI <span>Medical</span> Assistant</div>
        <div class="page-subtitle">Ask any health question — get structured, reliable answers instantly</div>
    </div>
    """, unsafe_allow_html=True)

    # Stats strip
    n_hist = len(history)
    st.markdown(f"""
    <div class="stats-strip">
        <div class="stat-card">
            <div class="stat-number">{n_hist}</div>
            <div class="stat-label">Questions Asked</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">5</div>
            <div class="stat-label">Answer Sections</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">24/7</div>
            <div class="stat-label">Available</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Input area
    st.markdown('<div class="input-container">', unsafe_allow_html=True)

    question = st.text_input(
        "Your Question",
        placeholder="e.g. What are the symptoms of diabetes? How do I treat a fever?",
        label_visibility="collapsed",
        key="question_input"
    )

    col_btn, col_tip = st.columns([1, 3])
    with col_btn:
        ask_clicked = st.button("Ask MediAI →", use_container_width=True)

    with col_tip:
        st.markdown("""
        <div style="font-size:12px;color:rgba(200,220,240,0.3);padding-top:14px;">
            ℹ️ For emergencies always contact a healthcare professional
        </div>
        """, unsafe_allow_html=True)

    # Quick examples (Updated with callback fix)
    st.markdown("""
    <div style="margin-top:16px;">
        <div style="font-size:11px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:rgba(200,220,240,0.25);margin-bottom:10px;">
            Quick Examples
        </div>
    </div>
    """, unsafe_allow_html=True)

    ex_cols = st.columns(4)
    examples = [
        "What is hypertension?",
        "Signs of vitamin D deficiency",
        "How to manage anxiety?",
        "Common cold vs flu?"
    ]
    for idx, ex in enumerate(examples):
        with ex_cols[idx]:
            # Use on_click parameter to update state safely
            st.button(
                ex, 
                key=f"ex_{idx}", 
                use_container_width=True, 
                on_click=set_example, 
                args=(ex,)
            )

    st.markdown('</div>', unsafe_allow_html=True)

    # ===== ASK AI =====
    # Evaluate if button was clicked OR if a quick example triggered the question
    should_ask = ask_clicked or st.session_state.example_trigger

    if should_ask and question and question.strip():
        # Reset the trigger so it doesn't loop
        st.session_state.example_trigger = False 
        
        st.session_state.last_question = question.strip()

        with st.spinner("🧠 Analyzing your question..."):
            try:
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {
                            "role": "system",
                            "content": """You are MediAI, a professional medical assistant AI.

Always answer in this exact structured format:

**1. Definition**
Explain what the condition/topic is in 2-3 clear sentences.

**2. Causes**
List the main causes or risk factors.

**3. Symptoms**
List common symptoms clearly.

**4. Prevention**
Give actionable prevention tips.

**5. When to See a Doctor**
Clearly state warning signs that require immediate medical attention.

Be concise, accurate, and compassionate. Use plain language.
Remind the user to consult a doctor for personal diagnosis."""
                        },
                        {"role": "user", "content": question.strip()}
                    ],
                    max_tokens=1200,
                    temperature=0.4,
                )

                answer = response.choices[0].message.content
                st.session_state.last_answer = answer
                save_history(user_id, question.strip(), answer)
                st.session_state.selected_q = ""
                st.session_state.selected_a = ""

            except Exception as e:
                # Upgraded to use traceback for detailed error logging
                error_details = traceback.format_exc()
                logging.error(f"Groq API Error:\n{error_details}")
                st.error("⚠️ Could not get a response. Please try again.")

    # ===== SHOW ANSWER =====
    if st.session_state.last_answer:
        q_display = st.session_state.last_question[:60] + "…" if len(st.session_state.last_question) > 60 else st.session_state.last_question
        answer_safe = st.session_state.last_answer.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")

        st.markdown(f"""
        <div class="answer-card">
            <div class="answer-header">
                <div class="answer-header-icon">🧠</div>
                <div class="answer-header-title">AI Medical Analysis</div>
                <div class="answer-header-q">{q_display}</div>
            </div>
            <div class="answer-body">{answer_safe}</div>
            <div style="margin-top:20px;padding-top:16px;border-top:1px solid rgba(0,200,180,0.08);
                font-size:11px;color:rgba(200,220,240,0.28);">
                ⚠️ This is AI-generated information, not a medical diagnosis. Consult a licensed doctor.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ===== SELECTED HISTORY =====
    elif st.session_state.get("selected_q"):
        ans_safe = st.session_state.selected_a.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        st.markdown(f"""
        <div class="selected-hist-card">
            <div class="selected-hist-label">📌 From History</div>
            <div class="selected-hist-q">{st.session_state.selected_q}</div>
            <div class="selected-hist-a">{ans_safe}</div>
        </div>
        """, unsafe_allow_html=True)

    # ===== ADMIN PANEL =====
    if role == "admin":
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        all_data = get_all_history()

        st.markdown(f"""
        <div class="admin-banner">
            <div class="admin-banner-icon">🔐</div>
            <div>
                <div class="admin-banner-title">Admin Dashboard</div>
                <div class="admin-banner-sub">{len(all_data)} total queries across all users</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        search_user = st.text_input("Filter by username", placeholder="🔍  Search users...", key="admin_search")

        for u, q, a in all_data:
            if search_user.lower() in u.lower():
                a_preview = a[:120] + "…" if len(a) > 120 else a
                a_safe = a_preview.replace("<", "&lt;").replace(">", "&gt;")
                q_safe = q.replace("<", "&lt;").replace(">", "&gt;")
                st.markdown(f"""
                <div class="admin-row">
                    <div class="admin-row-user">👤 {u}</div>
                    <div class="admin-row-q">❓ {q_safe}</div>
                    <div class="admin-row-a">{a_safe}</div>
                </div>
                """, unsafe_allow_html=True)