import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "assets" / "static" / "db" / "anomaly_dashboard.db"

_SEED_COMPONENTS = [
    ("cable", "Cable"),
    ("grid", "Grid"),
    ("metal_nut", "Metal Nut"),
    ("screw", "Screw"),
    ("transistor", "Transistor"),
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS components (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    key          TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS instances (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    component_id INTEGER NOT NULL REFERENCES components(id) ON DELETE CASCADE,
    label        TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    UNIQUE(component_id, label)
);

CREATE TABLE IF NOT EXISTS inspections (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    instance_id      INTEGER NOT NULL REFERENCES instances(id) ON DELETE CASCADE,
    timestamp        TEXT NOT NULL,
    error_percentage REAL NOT NULL,
    raw_image_path   TEXT NOT NULL,
    heatmap_path     TEXT NOT NULL,
    is_critical      INTEGER NOT NULL CHECK (is_critical IN (0,1))
);

CREATE INDEX IF NOT EXISTS idx_inspections_instance_ts ON inspections(instance_id, timestamp);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

DEFAULT_CRITICAL_THRESHOLD = 15.0


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.executescript(_SCHEMA)
        conn.executemany(
            "INSERT OR IGNORE INTO components (key, display_name) VALUES (?, ?)",
            _SEED_COMPONENTS,
        )
        conn.execute(
            "INSERT OR IGNORE INTO settings (key, value) VALUES ('critical_threshold', ?)",
            (str(DEFAULT_CRITICAL_THRESHOLD),),
        )


def list_components() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM components ORDER BY id").fetchall()
        return [dict(r) for r in rows]


def get_component(component_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM components WHERE id = ?", (component_id,)).fetchone()
        return dict(row) if row else None


def list_instances(component_id: int | None = None) -> list[dict]:
    query = """
        SELECT i.*, c.key AS component_key, c.display_name AS component_display_name
        FROM instances i
        JOIN components c ON c.id = i.component_id
    """
    params: tuple = ()
    if component_id is not None:
        query += " WHERE i.component_id = ?"
        params = (component_id,)
    query += " ORDER BY i.id"
    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def get_instance(instance_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT i.*, c.key AS component_key, c.display_name AS component_display_name
            FROM instances i
            JOIN components c ON c.id = i.component_id
            WHERE i.id = ?
            """,
            (instance_id,),
        ).fetchone()
        return dict(row) if row else None


def create_instance(component_id: int, label: str) -> dict:
    if get_component(component_id) is None:
        raise ValueError("component not found")
    with _connect() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO instances (component_id, label, created_at) VALUES (?, ?, ?)",
                (component_id, label, datetime.now(timezone.utc).isoformat()),
            )
        except sqlite3.IntegrityError:
            raise ValueError("label already exists for this component")
        instance_id = cur.lastrowid
    return get_instance(instance_id)


def insert_inspection(
    instance_id: int,
    error_percentage: float,
    raw_image_path: str,
    heatmap_path: str,
    is_critical: bool,
    timestamp: str | None = None,
) -> dict:
    timestamp = timestamp or datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO inspections
                (instance_id, timestamp, error_percentage, raw_image_path, heatmap_path, is_critical)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (instance_id, timestamp, error_percentage, raw_image_path, heatmap_path, int(is_critical)),
        )
        inspection_id = cur.lastrowid
    return get_inspection(inspection_id)


def get_inspection(inspection_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT insp.*, i.label AS instance_label,
                   c.key AS component_key, c.display_name AS component_display_name
            FROM inspections insp
            JOIN instances i ON i.id = insp.instance_id
            JOIN components c ON c.id = i.component_id
            WHERE insp.id = ?
            """,
            (inspection_id,),
        ).fetchone()
        return dict(row) if row else None


def get_timeseries(instance_id: int) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, timestamp, error_percentage, is_critical
            FROM inspections
            WHERE instance_id = ?
            ORDER BY timestamp ASC
            """,
            (instance_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_comparison_series(instance_id_a: int, instance_id_b: int) -> dict:
    return {
        "a": {"instance": get_instance(instance_id_a), "series": get_timeseries(instance_id_a)},
        "b": {"instance": get_instance(instance_id_b), "series": get_timeseries(instance_id_b)},
    }


def get_critical_threshold() -> float:
    with _connect() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = 'critical_threshold'").fetchone()
        return float(row["value"]) if row else DEFAULT_CRITICAL_THRESHOLD


def set_critical_threshold(value: float) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO settings (key, value) VALUES ('critical_threshold', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (str(value),),
        )


def delete_inspection(inspection_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, raw_image_path, heatmap_path FROM inspections WHERE id = ?",
            (inspection_id,),
        ).fetchone()
        if row is None:
            return None
        conn.execute("DELETE FROM inspections WHERE id = ?", (inspection_id,))
    return dict(row)


def delete_inspections_for_instance(instance_id: int) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, raw_image_path, heatmap_path FROM inspections WHERE instance_id = ?",
            (instance_id,),
        ).fetchall()
        conn.execute("DELETE FROM inspections WHERE instance_id = ?", (instance_id,))
    return [dict(r) for r in rows]
