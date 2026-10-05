"""
CRRA Lab C2 - Mock Contract Management API

Serves data/contracts.csv on http://localhost:5001 with derived policy fields,
calculated against a fixed simulated date so every run gives identical results.

Run from the project root:
    python mcp_server/contract_shim.py
"""

import csv
from datetime import date, timedelta
from pathlib import Path

from flask import Flask, jsonify, request

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "contracts.csv"
SIMULATED_TODAY = date(2025, 4, 1)
EDITABLE_FIELDS = {"status", "owner", "proposed_uplift_pct"}

app = Flask(__name__)
app.json.sort_keys = False


def notice_state(deadline: date, renewal: date) -> str:
    if renewal < SIMULATED_TODAY:
        return "EXPIRED"
    if deadline <= SIMULATED_TODAY:              # deadline passed, renewal not yet
        return "INSIDE_WINDOW"
    if (deadline - SIMULATED_TODAY).days <= 30:
        return "APPROACHING"
    return "OPEN"


def approval_band(value: int) -> str:
    if value < 1_000_000:
        return "A"
    return "B" if value <= 5_000_000 else "C"


def enrich(row: dict) -> dict:
    for key in ("annual_value_inr", "notice_days", "seats_purchased", "seats_active"):
        row[key] = int(row[key])
    row["proposed_uplift_pct"] = float(row["proposed_uplift_pct"])
    row["auto_renew"] = row["auto_renew"].strip().upper() == "Y"

    renewal = date.fromisoformat(row["renewal_date"])
    deadline = renewal - timedelta(days=row["notice_days"])
    row["notice_deadline"] = deadline.isoformat()
    row["days_to_renewal"] = (renewal - SIMULATED_TODAY).days
    row["notice_state"] = notice_state(deadline, renewal)
    row["utilisation_pct"] = (
        round(100 * row["seats_active"] / row["seats_purchased"], 1)
        if row["seats_purchased"] else None      # AMC / support: no seats
    )
    row["approval_band"] = approval_band(row["annual_value_inr"])
    return row


def load_contracts() -> dict[str, dict]:
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return {r["contract_id"].upper(): enrich(r) for r in csv.DictReader(f)}


CONTRACTS = load_contracts()                     # loaded once, at import time


def error(message: str, status: int):
    return jsonify({"error": message}), status


@app.get("/health")
def health():
    return jsonify({"status": "ok", "contracts_loaded": len(CONTRACTS),
                    "simulated_today": SIMULATED_TODAY.isoformat()})


@app.get("/api/contracts")
def list_contracts():
    results = list(CONTRACTS.values())
    for param, field in (("category", "category"), ("band", "approval_band"),
                         ("notice_state", "notice_state")):
        wanted = request.args.get(param)
        if wanted:
            results = [c for c in results if c[field].lower() == wanted.lower()]
    return jsonify({"count": len(results), "contracts": results})


@app.get("/api/contracts/expiring")
def expiring():
    try:
        days = int(request.args.get("days", 90))
    except ValueError:
        return error("days must be a whole number", 400)
    results = sorted((c for c in CONTRACTS.values() if 0 <= c["days_to_renewal"] <= days),
                     key=lambda c: c["days_to_renewal"])
    return jsonify({"window_days": days, "count": len(results), "contracts": results})


@app.get("/api/contracts/<contract_id>")
def get_contract(contract_id):
    contract = CONTRACTS.get(contract_id.upper())
    return jsonify(contract) if contract else error(f"Contract {contract_id} not found", 404)


@app.get("/api/categories")
def categories():
    grouped: dict[str, dict] = {}
    for c in CONTRACTS.values():
        g = grouped.setdefault(c["category"], {"category": c["category"], "vendor_count": 0,
                                               "total_annual_value_inr": 0, "vendors": []})
        g["vendor_count"] += 1
        g["total_annual_value_inr"] += c["annual_value_inr"]
        g["vendors"].append({k: c[k] for k in ("contract_id", "vendor", "annual_value_inr",
                                               "utilisation_pct")})
    result = sorted(grouped.values(), key=lambda g: g["category"])
    return jsonify({"count": len(result), "categories": result})


@app.patch("/api/contracts/<contract_id>")
def update_contract(contract_id):
    """In-memory only: restarting the server resets every change."""
    contract = CONTRACTS.get(contract_id.upper())
    if not contract:
        return error(f"Contract {contract_id} not found", 404)
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not payload:
        return error("Send a JSON object with status, owner or proposed_uplift_pct", 400)
    unknown = set(payload) - EDITABLE_FIELDS
    if unknown:
        return error(f"Cannot update: {', '.join(sorted(unknown))}", 400)
    if "proposed_uplift_pct" in payload:
        try:
            payload["proposed_uplift_pct"] = float(payload["proposed_uplift_pct"])
        except (TypeError, ValueError):
            return error("proposed_uplift_pct must be a number", 400)
    for field in ("status", "owner"):
        if field in payload and not (isinstance(payload[field], str) and payload[field].strip()):
            return error(f"{field} must be a non-empty string", 400)
    contract.update(payload)
    return jsonify({"updated": sorted(payload), "contract": contract})


if __name__ == "__main__":
    print(f"Mock Contract API: {len(CONTRACTS)} contracts, simulated today "
          f"{SIMULATED_TODAY}\n  http://localhost:5001/health")
    app.run(host="127.0.0.1", port=5001, debug=False)