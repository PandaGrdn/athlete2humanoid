"""Durable storage for a visitor's builds and trainings (docs/DEPLOY.md, "Keeping environments between visits").

Job state lives in memory and in a SQLite journal under A2H_DATA_DIR, and a live build's files sit with its upload
there; on Cloud Run that folder is /tmp, so a new revision or a restart forgets everything. A Store keeps what a
visitor made past that: a finished live build's files (processed.mp4, mask.mp4, live.json, task_spec.json, qpos.npy,
env.zip), a finished training's (train/weights.json, train/policy.pt, train/attempts/*) and the job's record (its
public JSON with a short log, who owns it, what was kept). JobStore (server/jobs.py) writes to it when a build or a
training ends, reads a job back when one that isn't in memory is asked for, and fetches its files into the job's
folder again before serving them (materialize). Raw uploads are never kept: they expire as before.

Two backends behind one interface, picked by A2H_STORE:
  unset               LocalStore: <A2H_DATA_DIR>/persist/jobs/<id>/<rel> and records.sqlite3 there (the default; tests)
  gcs://bucket[/pfx]  GcsStore: objects <pfx>/jobs/<id>/<rel>, the record at <pfx>/jobs/<id>/record.json, and an
                      owner index <pfx>/owners/<key>/<id> for "My environments" (one list_blobs by prefix); needs
                      google-cloud-storage and credentials (Cloud Run's service account: scripts/cloudrun_store.sh)

Ownership is a per-visitor key the browser makes on its first visit and sends as X-A2H-Owner (32 hex chars): no
accounts, and nobody can list another visitor's jobs without their key. A job's id stays the only key to its files.
"""

import json
import logging
import re
import shutil
import sqlite3
import threading
import time
from pathlib import Path

log = logging.getLogger("a2h.store")
OWNER_HEADER = "x-a2h-owner"
OWNER_KEY = re.compile(r"^[a-f0-9]{32}$")
REL = re.compile(r"^(?:[\w.-]+/)*[\w.-]+$")  # a kept file's path under the job: no '..', no leading slash
LIVE_FILES = ("processed.mp4", "mask.mp4", "live.json", "task_spec.json", "qpos.npy", "env.zip")
TRAIN_FILES = ("train/weights.json", "train/policy.pt", "train/attempts/index.json")
MAX_RECORD_LOGS = 60


def owner_key(headers):
    """The visitor's key from the request headers (an ASGI scope's list, or a Starlette Headers), or None."""
    items = headers.items() if hasattr(headers, "items") else [(k.decode("latin-1"), v.decode("latin-1")) for k, v in headers]
    for k, v in items:
        if k.lower() == OWNER_HEADER:
            v = v.strip().lower()
            return v if OWNER_KEY.match(v) else None
    return None


def kept_files(folder):
    """The files under a live job's folder worth keeping: the build's, and a training's attempts."""
    folder = Path(folder)
    out = [f for f in LIVE_FILES + TRAIN_FILES if (folder / f).is_file()]
    att = folder / "train" / "attempts"
    if att.is_dir():
        out += sorted(f"train/attempts/{p.name}" for p in att.iterdir() if re.match(r"^iter_\d{4,6}\.npz$", p.name))
    return out


def store_for(settings):
    """The Store A2H_STORE names: LocalStore by default."""
    url = getattr(settings, "store_url", "") or ""
    if url.startswith("gcs://"):
        bucket, _, prefix = url[6:].partition("/")
        return GcsStore(bucket, prefix.strip("/"))
    return LocalStore(settings.data_dir / "persist")


class Store:
    """What the backends implement. `rel` paths are relative to the job (REL); records are JSON dicts."""

    kind = "none"

    def put(self, jid, rel, path):
        raise NotImplementedError

    def fetch(self, jid, rel, dest):
        """-> True when the file was written to dest."""
        raise NotImplementedError

    def files(self, jid):
        raise NotImplementedError

    def delete(self, jid):
        raise NotImplementedError

    def save_record(self, jid, owner, record):
        raise NotImplementedError

    def get_record(self, jid):
        raise NotImplementedError

    def records_for(self, owner):
        """-> [record], newest first."""
        raise NotImplementedError

    def delete_record(self, jid):
        raise NotImplementedError

    # --- shared ----------------------------------------------------------------------------------
    def keep(self, jid, folder, rels=None):
        """Put a job folder's kept files; -> the rels kept."""
        done = []
        for rel in (rels or kept_files(folder)):
            if REL.match(rel) and (Path(folder) / rel).is_file():
                self.put(jid, rel, Path(folder) / rel)
                done.append(rel)
        return done

    def materialize(self, jid, folder, rels):
        """Fetch a job's kept files into its folder again (after a restart); -> how many came."""
        n = 0
        for rel in rels:
            if not REL.match(rel):
                continue
            dest = Path(folder) / rel
            if dest.is_file():
                n += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                if self.fetch(jid, rel, dest):
                    n += 1
            except Exception:  # noqa: BLE001
                log.exception("materialize failed", extra={"fields": {"job": jid, "rel": rel}})
        return n

    def ping(self):
        return True


class LocalStore(Store):
    kind = "local"

    def __init__(self, root):
        self.root = Path(root)
        (self.root / "jobs").mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(str(self.root / "records.sqlite3"), check_same_thread=False, isolation_level=None, timeout=10)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, owner TEXT, updated_at REAL NOT NULL, body TEXT NOT NULL)")
        self.db.execute("CREATE INDEX IF NOT EXISTS records_owner ON records (owner, updated_at)")

    def _path(self, jid, rel):
        return self.root / "jobs" / jid / rel

    def put(self, jid, rel, path):
        dest = self._path(jid, rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        shutil.copyfile(path, tmp)
        tmp.replace(dest)

    def fetch(self, jid, rel, dest):
        src = self._path(jid, rel)
        if not src.is_file():
            return False
        shutil.copyfile(src, dest)
        return True

    def files(self, jid):
        d = self.root / "jobs" / jid
        return sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()) if d.is_dir() else []

    def delete(self, jid):
        shutil.rmtree(self.root / "jobs" / jid, ignore_errors=True)
        self.delete_record(jid)

    def save_record(self, jid, owner, record):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO records VALUES (?, ?, ?, ?)", (jid, owner, time.time(), json.dumps(record)))

    def get_record(self, jid):
        with self.lock:
            row = self.db.execute("SELECT body FROM records WHERE id = ?", (jid,)).fetchone()
        try:
            return json.loads(row[0]) if row else None
        except ValueError:
            return None

    def records_for(self, owner):
        with self.lock:
            rows = self.db.execute("SELECT body FROM records WHERE owner = ? ORDER BY updated_at DESC", (owner,)).fetchall()
        out = []
        for (body,) in rows:
            try:
                out.append(json.loads(body))
            except ValueError:
                continue
        return out

    def delete_record(self, jid):
        with self.lock:
            self.db.execute("DELETE FROM records WHERE id = ?", (jid,))

    def ping(self):
        with self.lock:
            return self.db.execute("SELECT 1").fetchone() == (1,)

    def close(self):
        with self.lock:
            self.db.close()


class GcsStore(Store):
    """A Google Cloud Storage bucket. `client` is google.cloud.storage.Client() unless a stand-in is given (tests)."""

    kind = "gcs"

    def __init__(self, bucket, prefix="", client=None):
        self.bucket_name, self.prefix = bucket, (prefix + "/" if prefix else "")
        self._client, self._bucket = client, None

    def bucket(self):
        if self._bucket is None:
            if self._client is None:
                from google.cloud import storage  # imported here: hosts without the GCS backend never need it
                self._client = storage.Client()
            self._bucket = self._client.bucket(self.bucket_name)
        return self._bucket

    def _obj(self, jid, rel):
        return f"{self.prefix}jobs/{jid}/{rel}"

    def put(self, jid, rel, path):
        self.bucket().blob(self._obj(jid, rel)).upload_from_filename(str(path))

    def fetch(self, jid, rel, dest):
        b = self.bucket().blob(self._obj(jid, rel))
        if not b.exists():
            return False
        tmp = Path(dest).with_suffix(Path(dest).suffix + ".part")
        b.download_to_filename(str(tmp))
        tmp.replace(dest)
        return True

    def files(self, jid):
        p = self._obj(jid, "")
        return sorted(b.name[len(p):] for b in self.bucket().list_blobs(prefix=p) if b.name != p + "record.json")

    def delete(self, jid):
        rec = self.get_record(jid)
        for b in list(self.bucket().list_blobs(prefix=self._obj(jid, ""))):
            b.delete()
        if rec and rec.get("owner"):
            b = self.bucket().blob(f"{self.prefix}owners/{rec['owner']}/{jid}")
            if b.exists():
                b.delete()

    def save_record(self, jid, owner, record):
        record = dict(record, owner=owner) if owner else record  # delete() finds the owner index from the record
        self.bucket().blob(self._obj(jid, "record.json")).upload_from_string(json.dumps(record), content_type="application/json")
        if owner:
            self.bucket().blob(f"{self.prefix}owners/{owner}/{jid}").upload_from_string(str(record.get("updated_at", "")))

    def get_record(self, jid):
        b = self.bucket().blob(self._obj(jid, "record.json"))
        if not b.exists():
            return None
        try:
            return json.loads(b.download_as_bytes())
        except ValueError:
            return None

    def records_for(self, owner):
        p = f"{self.prefix}owners/{owner}/"
        ids = [b.name[len(p):] for b in self.bucket().list_blobs(prefix=p)]
        out = [r for r in (self.get_record(j) for j in ids if OWNER_KEY.match(j) or re.fullmatch(r"[a-f0-9]{12,32}", j)) if r]
        out.sort(key=lambda r: r.get("updated_at", 0), reverse=True)
        return out

    def delete_record(self, jid):
        b = self.bucket().blob(self._obj(jid, "record.json"))
        if b.exists():
            b.delete()

    def ping(self):
        try:
            return self.bucket().exists()
        except Exception:  # noqa: BLE001
            return False
