import sqlite3

def connect():
    return sqlite3.connect("database.db", check_same_thread=False)

def create_tables():
    conn = connect()
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        role TEXT DEFAULT 'user'
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        question TEXT,
        answer TEXT
    )
    """)

    conn.commit()
    conn.close()


def signup(username, password):
    conn = connect()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()


def login(username, password):
    conn = connect()
    c = conn.cursor()

    c.execute("SELECT id, role FROM users WHERE username=? AND password=?", (username, password))
    user = c.fetchone()

    conn.close()
    return user


def save_history(user_id, q, a):
    conn = connect()
    c = conn.cursor()
    c.execute("INSERT INTO history (user_id, question, answer) VALUES (?, ?, ?)", (user_id, q, a))
    conn.commit()
    conn.close()


def get_user_history(user_id):
    conn = connect()
    c = conn.cursor()
    c.execute("SELECT question, answer FROM history WHERE user_id=?", (user_id,))
    data = c.fetchall()
    conn.close()
    return data


def clear_user_history(user_id):
    conn = connect()
    c = conn.cursor()
    c.execute("DELETE FROM history WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()


def get_all_history():
    conn = connect()
    c = conn.cursor()
    c.execute("""
    SELECT users.username, history.question, history.answer
    FROM history
    JOIN users ON users.id = history.user_id
    """)
    data = c.fetchall()
    conn.close()
    return data