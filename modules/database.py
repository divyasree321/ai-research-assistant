import sqlite3

DB_PATH = "data/search_history.db"

def init_db():

    conn = sqlite3.connect(DB_PATH)

    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS searches(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT,
        answer TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


def save_search(question, answer):

    conn = sqlite3.connect(DB_PATH)

    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO searches(question, answer)
        VALUES (?,?)
        """,
        (question, answer)
    )

    conn.commit()
    conn.close()


def get_recent_searches(limit=5):

    conn = sqlite3.connect(DB_PATH)

    cur = conn.cursor()

    cur.execute(
        """
        SELECT question
        FROM searches
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,)
    )

    rows = cur.fetchall()

    conn.close()

    return [r[0] for r in rows]