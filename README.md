🩺 MediAI – AI Medical Assistant
MediAI is a modern AI-powered medical assistant built using Streamlit and Groq API.
It allows users to ask health-related questions and receive structured, easy-to-understand medical information.

🚀 Features
👤 User Features

🔐 Secure Login & Signup system
💬 Ask medical questions using AI
📜 Personal search history
🔍 Filter/search previous questions
🧹 Clear chat & delete personal data
📄 Upload PDF medical reports (basic support)


🤖 AI Capabilities

Structured responses:

Definition
Causes
Symptoms
Prevention
When to see a doctor


Fast responses using Groq API
Error handling for API failures


🧠 Tech Stack

Frontend: Streamlit
Backend: Python
Database: SQLite
AI API: Groq (LLaMA models)
Authentication: SHA256 hashing


📂 Project Structure
AI Medical Assistant/
│── app.py
│── database.db
│── requirements.txt
│── .env
│── utils/
│   └── db.py
│── app.log


⚙️ Installation
1. Clone the repository
git clone https://github.com/Eno1254/ai-medical-assistant.git
cd ai-medical-assistant


2. Install dependencies
pip install -r requirements.txt


3. Add API Key
Create a .env file:
GROQ_API_KEY=your_api_key_here


4. Run the app
streamlit run app.py


🔐 Security Notes

Passwords are hashed using SHA256
.env file is excluded from Git
User data stored locally (SQLite)


🧪 Week 8: Peer Testing & Iteration
✔ Actions Performed

Shared application with multiple users
Collected structured feedback using Google Form

❗ Issues Identified

Sidebar not functioning properly
Question suggestion feature causing UI conflicts
Minor usability issues

🔧 Fixes Implemented

Fixed sidebar navigation and interaction
Removed conflicting suggestion feature
Improved UI responsiveness and stability
Enhanced error handling


📊 Feedback Summary

Users found UI clean and modern
AI responses were helpful and structured
Some improvements needed in interaction flow (fixed)


⚠️ Disclaimer
This application provides AI-generated medical information.
It is not a substitute for professional medical advice.
Always consult a licensed doctor.

👨‍💻 Author
Enosh 
Computer Science Engineering Student
Lovely Professional University

📌 License
This project is for educational purposes.
