"""
CRRA Lab C4 - Audit Trail

Appends one JSON object per line to logs/audit_trail.jsonl and prints each entry.
The file is APPEND-ONLY by design: re-runs add to it rather than replacing it,
which is what an audit trail is for. Delete the file by hand for a clean demo.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "audit_trail.jsonl"


class AuditLogger:
    def __init__(self, log_path: Path = LOG_PATH, echo: bool = True):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self.entries: list[dict] = []

    def log(self, node: str, event: str, contract_id: str, actor: str = "system",
            **details) -> dict:
        """Record one event. Extra keyword arguments are stored under 'details'."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "node": node,
            "event": event,
            "contract_id": contract_id,
            "actor": actor,
            "details": details,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        self.entries.append(entry)
        if self.echo:
            print(f"  [AUDIT] {json.dumps(entry, ensure_ascii=False, default=str)}")
        return entry

    def summary(self) -> None:
        print(f"\n{'=' * 78}\nAUDIT TRAIL: {len(self.entries)} entries this run\n{'=' * 78}")
        print(f"{'Time (UTC)':<10}{'Contract':<10}{'Node':<14}{'Event':<22}Actor")
        print("-" * 78)
        for e in self.entries:
            print(f"{e['timestamp'][11:19]:<10}{e['contract_id']:<10}{e['node']:<14}"
                  f"{e['event']:<22}{e['actor']}")
        print("-" * 78)
        print(f"Appended to {self.log_path}")