from pathlib import Path
import os
import sys


def _user_data_dir() -> Path:
    """Return a writable per-user data directory on every supported OS."""
    if sys.platform == "win32":
        root = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA")
        base = Path(root or Path.home())
        return base / "Sequence-Detection"

    # XDG is standard on Ubuntu and other Linux desktop distributions.
    xdg = Path(os.getenv("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return xdg / "sequence-detection"


def get_desktop_dir() -> Path:
    """Resolve Desktop robustly on ANY PC, handling Windows Registry, OneDrive redirection, Linux, and custom user profiles."""
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders",
            )
            val, _ = winreg.QueryValueEx(key, "Desktop")
            winreg.CloseKey(key)
            expanded = os.path.expandvars(val)
            p = Path(expanded)
            if p.exists() and p.is_dir():
                return p
            p.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            pass

    home = Path.home()
    # Check OneDrive Desktop or Standard Desktop
    for candidate in [home / "OneDrive" / "Desktop", home / "Desktop"]:
        if candidate.exists() and candidate.is_dir():
            return candidate

    # Linux standard XDG desktop check
    xdg_desktop = os.getenv("XDG_DESKTOP_DIR")
    if xdg_desktop:
        try:
            p = Path(xdg_desktop)
            if p.exists() and p.is_dir():
                return p
        except Exception:
            pass

    # Ensure fallback Desktop exists on any PC
    try:
        fallback = home / "Desktop"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    except Exception:
        fallback = APP_DIR / "Desktop"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


APP_DIR = _user_data_dir()
APP_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_PATH = APP_DIR / "sequence_detection.db"
_DEFAULT_APP_DB = DATABASE_PATH


from datetime import datetime
import sqlite3
from typing import Optional


def _resolve_active_db_path() -> Optional[Path]:
    """Return the active SQLite DB file used by SQLAlchemy SessionLocal (or overridden DATABASE_PATH)."""
    if DATABASE_PATH != _DEFAULT_APP_DB and DATABASE_PATH.is_file():
        return DATABASE_PATH
    try:
        from app.config import settings
        db_url = getattr(settings, "DATABASE_URL", "") or ""
        if db_url.startswith("sqlite:///"):
            cand = Path(db_url[len("sqlite:///"):]).expanduser().resolve()
            if cand.is_file():
                return cand
    except Exception:
        pass
    if DATABASE_PATH.is_file():
        return DATABASE_PATH
    return None


def get_saved_output_root() -> Optional[Path]:
    """Return user-selected root output folder saved in SQLite training_settings, if configured."""
    db_path = _resolve_active_db_path()
    if not db_path:
        return None
    conn = None
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("SELECT value FROM training_settings WHERE key = 'output_root' LIMIT 1")
        row = cur.fetchone()
        if row and row[0] and str(row[0]).strip():
            p = Path(str(row[0]).strip()).expanduser().resolve()
            if p.name.lower() in {"sequence_train", "sequence_inference"}:
                p = p.parent
            return p
    except Exception:
        pass
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    return None


def get_inference_base_dir() -> Path:
    """Return '<saved_output_root>/sequence_inference' (or APP_DIR/sequence_inference fallback)."""
    saved_root = get_saved_output_root()
    base = (saved_root / "sequence_inference") if saved_root else (APP_DIR / "sequence_inference")
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_ml_output_dir() -> Path:
    """Return the daily output directory inside sequence_inference for ML inference sessions."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    out = get_inference_base_dir() / today_str
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_ml_session_config_dir() -> Path:
    """Return a guaranteed WRITABLE directory for temporary camera and video YAML configs."""
    cfg_dir = APP_DIR / "sessions"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return cfg_dir


def get_logs_dir() -> Path:
    """Return logs directory: project root 'logs' during development, APP_DIR 'logs' in production/frozen."""
    if getattr(sys, "frozen", False):
        log_dir = APP_DIR / "logs"
    else:
        dev_dir = Path(__file__).resolve().parent.parent.parent / "logs"
        try:
            dev_dir.mkdir(parents=True, exist_ok=True)
            return dev_dir
        except Exception:
            log_dir = APP_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

