# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

import contextlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

try:
    import fcntl  # POSIX; absent on Windows
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

SCHEMA_DIR = Path(__file__).parent / "schema"

PRAGMAS = [
    # journal_mode = DELETE: NO permanent sidecar files in the user's project
    # folder — the rollback journal exists only during a write transaction and
    # is unlinked at commit (WAL leaves `-wal`/`-shm`; SQLite can't relocate
    # them, and a Ctrl-C/crash hard-exit leaves them stranded — proven in the
    # wild). The GUI freezes that WAL was meant to fix are instead solved by
    # never blocking the GUI thread on a contended read: the thumb-cache reader
    # uses a tiny busy_timeout and falls through to the async loader on lock
    # (see MainWindow.ThumbLoader), so a large reprocess can hold the writer
    # without ever stalling the event loop. Opening a WAL-leftover `.agl` here
    # also folds it back + removes the sidecars (journal_mode switch checkpoints).
    "PRAGMA journal_mode = DELETE;",
    "PRAGMA synchronous = NORMAL;",
    "PRAGMA foreign_keys = ON;",
    "PRAGMA temp_store = MEMORY;",
    "PRAGMA cache_size = -64000;",
    "PRAGMA mmap_size = 268435456;",
    # Workers retry on contention. journal_mode=DELETE recreates+unlinks the
    # rollback journal per write; on Windows those file syscalls (plus AV
    # scanning the -journal) make each write much slower, so a burst of
    # contended writes can serialize past a short timeout → "database is
    # locked". 20s gives ample headroom (the GUI read path uses its own tiny
    # timeout + async fallback, so this never stalls the event loop).
    "PRAGMA busy_timeout = 20000;",
]


def execute_write(conn: sqlite3.Connection, sql: str, params: Iterable = (),
                  *, attempts: int = 6):
    """Run a write statement, retrying on a transient "database is locked".

    busy_timeout already retries lock *acquisition*, but on Windows a write
    can still surface SQLITE_BUSY when the journal file is momentarily held
    (AV / another connection unlinking it). A short bounded backoff on top of
    the PRAGMA timeout makes the hot insert path robust under contention
    without abandoning the single-file (DELETE-mode) design. Returns the
    cursor."""
    import time
    delay = 0.05
    for i in range(attempts):
        try:
            return conn.execute(sql, tuple(params))
        except sqlite3.OperationalError as e:
            if "locked" not in str(e).lower() or i == attempts - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 1.0)


def open_db(path: str | Path) -> sqlite3.Connection:
    """Open or create a project SQLite file, apply PRAGMAs, ensure schema."""
    path = str(path)
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    for stmt in PRAGMAS:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            # In-memory or shared-cache DBs reject some pragmas (e.g. WAL on :memory:).
            pass
    _migrate_serialized(conn, path)
    return conn


def compact_db(path: str | Path) -> None:
    """Fold the WAL back into the `.agl` and remove the `-wal`/`-shm`
    sidecars, leaving a single clean file at rest.

    Call on a clean shutdown AFTER all worker processes are reaped (so this
    is the only connection). WAL gives us concurrent reads while the app
    runs; switching journal_mode to DELETE here checkpoints any pending WAL
    frames into the main DB and deletes both sidecar files — so the user's
    project folder shows only `myproject.agl` when nothing is running. The
    next open re-enables WAL via the PRAGMA list. A crash / hard-exit skips
    this (the sidecars linger but SQLite recovers them on next open, and the
    following clean exit compacts them away). Best-effort: never raises."""
    p = str(path)
    if not p or p == ":memory:":
        return
    try:
        conn = sqlite3.connect(p, isolation_level=None, check_same_thread=False)
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            # journal_mode=DELETE removes the -wal/-shm files (needs to be the
            # sole connection; harmless no-op if the DB is already non-WAL).
            conn.execute("PRAGMA journal_mode = DELETE;")
        finally:
            conn.close()
    except Exception:
        pass


def _schema_current(conn: sqlite3.Connection) -> bool:
    """True when every migration file is already recorded as applied — the
    lock-free fast path for the common (already-migrated) open."""
    try:
        n = conn.execute("SELECT count(*) FROM _schema_migrations").fetchone()[0]
    except sqlite3.OperationalError:
        return False  # ledger table missing → never migrated
    return int(n) >= len(list(SCHEMA_DIR.glob("*.sql")))


def _migrate_serialized(conn: sqlite3.Connection, path: str) -> None:
    """Run `ensure_schema` under a cross-process advisory lock.

    Migrations include a non-idempotent table rebuild (0006). Without
    serialisation, the first open of a pre-migration project by the GUI +
    each spawned worker races: a second process running the rebuild hits
    "database is locked" (not a swallowed "duplicate column"), so its
    `open_db` raises and that worker dies on spawn — leaving its scan's
    spinner stuck. An advisory `flock` on a sidecar lock file lets exactly
    one opener apply the migration while the rest wait, then skip via the
    ledger. (The lock is a SEPARATE file, not the DB — flock on the DB file
    itself conflicts with SQLite's own locking on macOS.) Fast path skips
    the lock entirely once everything is applied.
    """
    if _schema_current(conn):
        return
    if fcntl is None or path == ":memory:" or not path:
        ensure_schema(conn)  # Windows / no fcntl / in-memory — unserialised
        return
    import hashlib
    import os
    import tempfile
    # Lock lives in the temp dir (keyed by the DB's absolute path) so it
    # doesn't litter a sidecar next to every project file.
    key = hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:16]
    lock_path = os.path.join(tempfile.gettempdir(), f"aglaia-migrate-{key}.lock")
    try:
        lockf = open(lock_path, "w")
    except OSError:
        ensure_schema(conn)  # can't create lock — best effort
        return
    try:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        ensure_schema(conn)  # re-reads the ledger inside the lock
    finally:
        try:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        lockf.close()


def ensure_schema(conn: sqlite3.Connection) -> None:
    # Applied-migration ledger. Additive migrations (ADD COLUMN, idempotent
    # UPDATEs, CREATE … IF NOT EXISTS) are safe to re-run, but a table
    # rebuild (e.g. 0006 relaxing a NOT NULL) is NOT — recording each file
    # once it succeeds means such rebuilds run exactly once per DB.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS _schema_migrations ("
        "filename TEXT PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    applied = {r[0] for r in conn.execute(
        "SELECT filename FROM _schema_migrations")}
    files = sorted(SCHEMA_DIR.glob("*.sql"))
    if not files:
        # Zero migrations = the schema dir didn't ship (packaging bug). Fail
        # loudly instead of leaving an empty, table-less DB behind.
        raise RuntimeError(
            f"No schema migrations found in {SCHEMA_DIR} — the bundle is "
            "missing aglaia/storage/schema/*.sql.")
    for f in files:
        if f.name in applied:
            continue
        sql = f.read_text(encoding="utf-8")
        try:
            conn.executescript(sql)
        except sqlite3.OperationalError as e:
            # `ALTER TABLE … ADD COLUMN` (used by additive migrations)
            # is not idempotent in SQLite < 3.35. If we hit "duplicate
            # column name", assume the migration was already applied to
            # this DB and move on.
            if "duplicate column name" not in str(e).lower():
                raise
        conn.execute(
            "INSERT OR IGNORE INTO _schema_migrations(filename, applied_at) "
            "VALUES (?, ?)",
            (f.name, datetime.now(timezone.utc).isoformat()),
        )


@contextlib.contextmanager
def db_session(path: str | Path) -> Iterator[sqlite3.Connection]:
    """Open a project DB, yield the connection, close on exit (incl. exceptions).

    Replaces the open_db + try/finally close pattern scattered across the
    GUI and web routes. Use as `with db_session(path) as conn: ...` or via
    FastAPI `Depends(get_conn)`.
    """
    conn = open_db(path)
    try:
        yield conn
    finally:
        conn.close()


def in_transaction(conn: sqlite3.Connection):
    """Context manager wrapping BEGIN IMMEDIATE/COMMIT, rolling back on
    exception. Nested-safe: if the connection is already inside a
    transaction, becomes a no-op so callers can compose freely.

    BEGIN IMMEDIATE takes the write lock up front — under multi-worker
    contention this fails fast at BEGIN (retried by busy_timeout)
    instead of deadlocking on a later lock upgrade."""
    class _Tx:
        def __enter__(self_inner):
            self_inner.owns = not conn.in_transaction
            if self_inner.owns:
                conn.execute("BEGIN IMMEDIATE")
            return conn

        def __exit__(self_inner, exc_type, exc, tb):
            if self_inner.owns:
                if exc_type is None:
                    conn.execute("COMMIT")
                else:
                    conn.execute("ROLLBACK")
            return False
    return _Tx()
