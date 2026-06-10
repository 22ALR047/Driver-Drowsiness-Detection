import os
import sqlite3
import hashlib
import uuid

DB_PATH = os.path.join(os.path.dirname(__file__), "drowsishield.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create Users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL
    )
    """)
    
    # Create Sessions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        timestamp TEXT NOT NULL,
        duration_sec INTEGER NOT NULL,
        blinks_count INTEGER NOT NULL,
        yawns_count INTEGER NOT NULL,
        max_fatigue REAL NOT NULL,
        status TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    """)
    
    conn.commit()
    conn.close()
    print(f"[INFO] SQLite database initialized at '{DB_PATH}'")

def hash_password(password, salt=None):
    """Secure password hashing using PBKDF2 with SHA256."""
    if salt is None:
        salt = uuid.uuid4().hex
    pwd_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000
    ).hex()
    return pwd_hash, salt

def create_user(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    pwd_hash, salt = hash_password(password)
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
            (username.strip().lower(), pwd_hash, salt)
        )
        conn.commit()
        return True, cursor.lastrowid
    except sqlite3.IntegrityError:
        return False, "Username already exists"
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, password_hash, salt FROM users WHERE username = ?",
        (username.strip().lower(),)
    )
    user = cursor.fetchone()
    conn.close()
    
    if user:
        pwd_hash, _ = hash_password(password, user['salt'])
        if pwd_hash == user['password_hash']:
            return True, user['id']
    return False, "Invalid username or password"

def log_session(user_id, timestamp, duration_sec, blinks_count, yawns_count, max_fatigue, status):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO sessions (user_id, timestamp, duration_sec, blinks_count, yawns_count, max_fatigue, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (user_id, timestamp, int(duration_sec), int(blinks_count), int(yawns_count), float(max_fatigue), status)
    )
    conn.commit()
    session_id = cursor.lastrowid
    conn.close()
    return session_id

def get_user_sessions(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, timestamp, duration_sec, blinks_count, yawns_count, max_fatigue, status FROM sessions WHERE user_id = ? ORDER BY id DESC",
        (user_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

# Initialize on import
init_db()
