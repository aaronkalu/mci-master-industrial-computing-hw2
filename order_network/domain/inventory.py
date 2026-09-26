import sqlite3
from contextlib import closing
from pathlib import Path

from order_network.models import OrderLine, StockLine, StockReport

SCHEMA = """
CREATE TABLE IF NOT EXISTS stock (
    material_number TEXT PRIMARY KEY,
    description     TEXT NOT NULL,
    on_hand         INTEGER NOT NULL CHECK (on_hand >= 0),
    reserved        INTEGER NOT NULL CHECK (reserved >= 0),
    warehouse       TEXT NOT NULL
);
"""

SEED = [
    ("149449", "Hydraulic pump housing", 40, 10, "Innsbruck"),
    ("255565", "Gearbox output shaft 40 mm", 500, 20, "Linz"),
    ("252653", "Flange coupling DN50", 8, 0, "Innsbruck"),
    ("252654", "Flange coupling DN65", 0, 0, "Innsbruck"),
    ("310022", "Servo drive mounting bracket", 150, 30, "Vienna"),
    ("418870", "Conveyor roller 500 mm", 1200, 200, "Linz"),
]


def init_db(db_path: Path, reset: bool = False) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if reset:
        db_path.unlink(missing_ok=True)
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT OR IGNORE INTO stock VALUES (?, ?, ?, ?, ?)", SEED)


def check_stock(db_path: Path, lines: list[OrderLine]) -> StockReport:
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        result = []
        for line in lines:
            row = conn.execute(
                "SELECT description, on_hand - reserved AS available, warehouse FROM stock WHERE material_number = ?",
                (line.material_number,),
            ).fetchone()
            available = max(row["available"], 0) if row else 0
            reserve = min(line.quantity, available)
            result.append(
                StockLine(
                    material_number=line.material_number,
                    description=row["description"] if row else None,
                    known_material=row is not None,
                    requested=line.quantity,
                    available=available,
                    reserve_now=reserve,
                    shortfall=line.quantity - reserve,
                    warehouse=row["warehouse"] if row else None,
                )
            )
    return StockReport(
        lines=result,
        fully_available=all(line.shortfall == 0 for line in result),
        production_request=[
            OrderLine(material_number=line.material_number, quantity=line.shortfall)
            for line in result if line.shortfall > 0
        ],
    )
