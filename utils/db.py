import sqlite3

DB_PATH = "database.db"

def connect():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def create_tables():
    conn = connect()
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT DEFAULT 'user',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """)

    conn.commit()
    conn.close()


def signup(username: str, password: str) -> bool:
    """Create a new user. Password should already be hashed."""
    conn = connect()
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username.strip(), password)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def login(username: str, password: str):
    """Return (id, role) if credentials match, else None."""
    conn = connect()
    c = conn.cursor()
    c.execute(
        "SELECT id, role FROM users WHERE username=? AND password=?",
        (username.strip(), password)
    )
    user = c.fetchone()
    conn.close()
    return user


def save_history(user_id: int, question: str, answer: str):
    conn = connect()
    c = conn.cursor()
    c.execute(
        "INSERT INTO history (user_id, question, answer) VALUES (?, ?, ?)",
        (user_id, question, answer)
    )
    conn.commit()
    conn.close()


def get_user_history(user_id: int):
    """Return list of (question, answer) for a user."""
    conn = connect()
    c = conn.cursor()
    c.execute(
        "SELECT question, answer FROM history WHERE user_id=? ORDER BY created_at DESC",
        (user_id,)
    )
    data = c.fetchall()
    conn.close()
    return data


def clear_user_history(user_id: int):
    conn = connect()
    c = conn.cursor()
    c.execute("DELETE FROM history WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()


def get_all_history():
    """Admin: return (username, question, answer) for all users."""
    conn = connect()
    c = conn.cursor()
    c.execute("""
        SELECT users.username, history.question, history.answer
        FROM history
        JOIN users ON users.id = history.user_id
        ORDER BY history.created_at DESC
    """)
    data = c.fetchall()
    conn.close()
    return data


def get_user_count() -> int:
    conn = connect()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users")
    count = c.fetchone()[0]
    conn.close()
    return count


def delete_user(user_id: int):
    conn = connect()
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
