"""Job state on disk: one SQLite row per job (the job's public JSON), so a restart doesn't lose jobs.

SQLite in WAL mode, one connection guarded by a lock: the app is one process, and writes are a few per
second at most (JobStore batches log-line updates). Rows are the source of truth only at startup; while
the process runs, the in-memory JobStore is.
"""

import json
import sqlite3
import threading

SCHEMA = """CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, status TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL,
    owner TEXT, body TEXT NOT NULL)"""


class Journal:
    def __init__(self, path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None, timeout=10)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute(SCHEMA)

    def save(self, rows):
        """rows: [(id, status, created_at, updated_at, owner, body dict)]."""
        if not rows:
            return
        with self.lock:
            self.db.execute("BEGIN")
            try:
                self.db.executemany("INSERT OR REPLACE INTO jobs VALUES (?, ?, ?, ?, ?, ?)",
                                    [(*r[:5], json.dumps(r[5])) for r in rows])
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise

    def load(self):
        with self.lock:
            rows = self.db.execute("SELECT owner, body FROM jobs ORDER BY created_at").fetchall()
        out = []
        for owner, body in rows:
            try:
                out.append((owner, json.loads(body)))
            except ValueError:
                continue
        return out

    def delete(self, ids):
        if ids:
            with self.lock:
                self.db.executemany("DELETE FROM jobs WHERE id = ?", [(i,) for i in ids])

    def ping(self):
        with self.lock:
            return self.db.execute("SELECT 1").fetchone() == (1,)

    def close(self):
        with self.lock:
            self.db.close()
