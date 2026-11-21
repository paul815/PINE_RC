"""Graceful shutdown utilities for the PINE backend."""

import logging
import os

log = logging.getLogger(__name__)

# The app whose database a shutdown flushes. Every caller of graceful_exit runs
# on a bare daemon thread started from a request, where Flask's application
# context is already gone — so ``db.engine`` raised "Working outside of
# application context", the warning went to the log on every quit, and the
# WAL checkpoint below never once ran. create_app() registers the app here.
_app = None


def register_app(app) -> None:
    """Remember the app so a shutdown from any thread can reach its database."""
    global _app
    _app = app


def flush_database(app=None) -> bool:
    """Checkpoint the SQLite WAL into the main file and dispose the engine.

    Nothing is lost without it — committed rows sit in ``pine.db-wal`` and
    SQLite replays them on the next open — but until then ``pine.db`` alone is
    behind: a copy of just that file misses them. Returns True when it ran.
    """
    from flask import current_app, has_app_context

    from .extensions import db

    if app is None:
        app = current_app._get_current_object() if has_app_context() else _app
    if app is None:
        log.warning('Could not flush SQLite during shutdown: no app registered')
        return False
    try:
        with app.app_context():
            with db.engine.connect() as conn:
                conn.execute(db.text("PRAGMA wal_checkpoint(TRUNCATE);"))
                log.info("SQLite WAL checkpoint completed")
            db.engine.dispose()
            log.info("Database engine disposed")
        return True
    except Exception:
        log.warning("Could not flush SQLite during shutdown", exc_info=True)
        return False


def graceful_exit(code: int = 0) -> None:
    """Flush SQLite WAL and dispose engine before terminating the process.

    Called from daemon threads where ``sys.exit()`` only raises SystemExit
    inside the thread (it does NOT terminate the process).  We therefore still
    call ``os._exit()`` — but only *after* flushing pending writes.
    """
    flush_database()
    os._exit(code)
