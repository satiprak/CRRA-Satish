"""
CRRA Lab C4 - Portfolio Orchestrator with a Human Approval Gate

    analysis -> policy_check -> (hitl if required) -> report

The model may recommend; plain Python decides whether a human must sign off,
and nothing is committed without that sign-off when policy demands it.

Prerequisites (run from the project root):
    1. python data/kb_setup.py               Lab C1 - builds the policy KB
    2. python mcp_server/contract_shim.py    Lab C2 - leave running in a 2nd terminal
    3. .env containing ANTHROPIC_API_KEY=...

Then:
    python orchestrator/supervisor.py                 (default portfolio)
    python orchestrator/supervisor.py CTR-1004 ...    (any contract IDs)
"""

import sys
from pathlib import Path

# `python orchestrator/supervisor.py` puts only orchestrator/ on the import path.
# Add the project root first so `guardrails` (and `data`) can be imported.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json                                        # noqa: E402
import os                                          # noqa: E402
from typing import TypedDict                       # noqa: E402

import anthropic                                   # noqa: E402
import chromadb                                    # noqa: E402
import requests                                    # noqa: E402
from dotenv import load_dotenv                     # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402

from guardrails.audit_logger import AuditLogger    # noqa: E402

load_dotenv(ROOT / ".env", override=True)          # .env beats a stale Windows variable

MODEL = "claude-opus-5"
EFFORT = "medium"                # temperature is not supported on this model
MAX_TOKENS = 4096
MAX_ROUNDS = 5
CONTRACT_API = "http://localhost:5001"
KB_COLLECTION = "crra_policy"
CHROMA_DIR = ROOT / "data" / "chroma_db"           # written by Lab C1
RECOMMENDATIONS = ["RENEW", "RENEGOTIATE", "CONSOLIDATE", "TERMINATE"]
CONFIDENCE_LEVELS = ["HIGH", "MEDIUM", "LOW"]
NO_DECISION = "NO_RECOMMENDATION"

# CTR-1010 has no policy trigger and should go straight to report: that contrast
# is the point. CTR-1012 (Band B, inside window) and CTR-1006 (no owner) must stop.
DEFAULT_PORTFOLIO = ["CTR-1010", "CTR-1012", "CTR-1006"]

audit = AuditLogger()
_client = None
_kb = None


# ======================================================================
# State
# ======================================================================

class ContractState(TypedDict):
    contract_id: str
    contract: dict
    recommendation: str
    confidence: str
    rationale: str
    policy_citation: str
    estimated_annual_impact_inr: int
    hitl_required: bool
    hitl_reason: str
    hitl_approved: bool
    approver: str
    final_status: str


def initial_state(contract_id: str) -> ContractState:
    return {"contract_id": contract_id, "contract": {}, "recommendation": "", "confidence": "",
            "rationale": "", "policy_citation": "", "estimated_annual_impact_inr": 0,
            "hitl_required": False, "hitl_reason": "", "hitl_approved": False,
            "approver": "", "final_status": ""}


# ======================================================================
# Helpers
# ======================================================================

def extract_text(response) -> str:
    """First block with text; a thinking block may come first."""
    for block in response.content:
        if getattr(block, "text", None):
            return block.text.strip()
    return ""


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "").strip())
    return _client


def _chunk(text: str) -> list[tuple[str, str]]:
    """Same rule as Lab C1: one chunk per '## ' section, '# ' title skipped."""
    chunks, heading, body = [], None, []
    for line in text.splitlines() + ["## "]:
        if line.startswith("## "):
            if heading and "\n".join(body).strip():
                chunks.append((heading, "\n".join(body).strip()))
            heading, body = line[3:].strip(), []
        elif not line.startswith("# "):
            body.append(line)
    return chunks


def get_kb():
    """Open Lab C1's persistent KB, building it if missing."""
    global _kb
    if _kb is not None:
        return _kb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        _kb = client.get_collection(KB_COLLECTION)
        if _kb.count():
            return _kb
        client.delete_collection(KB_COLLECTION)
    except Exception:
        pass
    _kb = client.create_collection(KB_COLLECTION, metadata={"hnsw:space": "cosine"})
    ids, docs, metas = [], [], []
    for md in sorted((ROOT / "data" / "kb").glob("*.md")):
        for i, (heading, body) in enumerate(_chunk(md.read_text(encoding="utf-8"))):
            ids.append(f"{md.stem}::{i:02d}")
            docs.append(f"{heading}\n\n{body}")
            metas.append({"source": md.name, "heading": heading, "chunk_index": i})
    _kb.add(ids=ids, documents=docs, metadatas=metas)
    print(f"  (built policy KB: {len(ids)} chunks)")
    return _kb


def search_policy(query: str) -> list[dict]:
    """Best section from each of the top 2 policy files."""
    kb = get_kb()
    res = kb.query(query_texts=[query], n_results=kb.count())
    best: dict[str, dict] = {}
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        if meta["source"] not in best:
            best[meta["source"]] = {"source": meta["source"], "section": meta["heading"],
                                    "confidence": round(1 - dist, 2), "text": doc}
        if len(best) == 2:
            break
    return list(best.values())


def fetch_contract(contract_id: str) -> tuple[dict | None, str | None]:
    try:
        r = requests.get(f"{CONTRACT_API}/api/contracts/{contract_id}", timeout=10)
    except requests.exceptions.RequestException:
        return None, ("Contract API unreachable at http://localhost:5001. Start it with: "
                      "python mcp_server/contract_shim.py")
    if r.status_code == 404:
        return None, f"Contract {contract_id} not found"
    if not r.ok:
        return None, f"Contract API returned HTTP {r.status_code}"
    return r.json(), None


# ======================================================================
# Node 1 - Analysis (the only node that calls the model)
# ======================================================================

ANALYSIS_TOOLS = [
    {
        "name": "search_policy",
        "description": ("Search the procurement policy. Returns the best section from each of "
                        "the top 2 policy files with a 0-1 confidence; below about 0.3 means "
                        "the policy may not cover the question. Use at most twice."),
        "input_schema": {"type": "object",
                         "properties": {"query": {"type": "string"}},
                         "required": ["query"]},
    },
    {
        "name": "submit_recommendation",
        "description": "Record the final recommendation. Call exactly once, as the last step.",
        "input_schema": {
            "type": "object",
            "properties": {
                "recommendation": {"type": "string", "enum": RECOMMENDATIONS},
                "confidence": {"type": "string", "enum": CONFIDENCE_LEVELS},
                "rationale": {"type": "string",
                              "description": "Two or three sentences citing the specific numbers."},
                "policy_citation": {"type": "string",
                                    "description": "File and section returned by search_policy, "
                                                   "e.g. 'renegotiation_levers.md §Price uplift benchmarks'"},
                "estimated_annual_impact_inr": {"type": "integer",
                                                "description": "Per year. 0 for RENEW on existing "
                                                               "terms; negative = saving."},
            },
            "required": ["recommendation", "confidence", "rationale", "policy_citation",
                         "estimated_annual_impact_inr"],
        },
    },
]

ANALYSIS_SYSTEM = """You are the Renewal Analysis Agent for Zensar BizOps. The contract's facts are
in the user message. Recommend exactly one of RENEW, RENEGOTIATE, CONSOLIDATE or TERMINATE.

Search the policy (at most twice) for the governing rule, then call submit_recommendation
exactly once. Your turn budget is small, so stop researching once you can make a call.

Guidance from policy:
- Utilisation above 85% with uplift of 8% or less is a healthy RENEW.
- Uplift above 15% is never accepted at first offer: RENEGOTIATE.
- Utilisation below 40% with an overlapping vendor in the same category: CONSOLIDATE.
- TERMINATE only if the capability itself is no longer needed. An UNASSIGNED owner is not
  evidence of that; absence of an owner is not absence of need.
- utilisation_pct of null means a support/AMC contract with no seats, not 0% usage.

Confidence: HIGH when figures and policy clearly agree; MEDIUM when the call rests on an
assumption you name; LOW when evidence is mixed, the policy match is weak, or a key fact is
missing. LOW is a valid and useful answer: it tells the human approver where to look. Give it
honestly rather than inventing certainty or searching again for a cleaner picture.

You do not decide whether a human must approve; a separate policy check does that.
Cite only a file and section that search_policy actually returned."""

FACT_FIELDS = ("contract_id", "vendor", "category", "business_unit", "owner", "annual_value_inr",
               "approval_band", "renewal_date", "notice_deadline", "notice_state",
               "days_to_renewal", "auto_renew", "seats_purchased", "seats_active",
               "utilisation_pct", "proposed_uplift_pct", "status")


def _validate(args: dict) -> str | None:
    if args.get("recommendation") not in RECOMMENDATIONS:
        return f"recommendation must be one of {RECOMMENDATIONS}"
    if args.get("confidence") not in CONFIDENCE_LEVELS:
        return f"confidence must be one of {CONFIDENCE_LEVELS}"
    for field in ("rationale", "policy_citation"):
        if not str(args.get(field, "")).strip():
            return f"{field} must not be empty"
    try:
        int(args.get("estimated_annual_impact_inr", 0))
    except (TypeError, ValueError):
        return "estimated_annual_impact_inr must be an integer"
    return None


def _no_decision(state: ContractState, contract: dict, reason: str) -> ContractState:
    """Never invent a recommendation. LOW confidence guarantees the HITL gate."""
    audit.log("analysis", "no_recommendation", state["contract_id"], reason=reason)
    print(f"  ⚠ {reason}")
    return {**state, "contract": contract, "recommendation": NO_DECISION, "confidence": "LOW",
            "rationale": reason, "policy_citation": "n/a", "estimated_annual_impact_inr": 0}


def analysis_node(state: ContractState) -> ContractState:
    cid = state["contract_id"]
    print(f"\n{'=' * 78}\nCONTRACT {cid}\n{'=' * 78}\n▶ ANALYSIS")

    contract, err = fetch_contract(cid)
    if err:
        print(f"  ✗ {err}")
        audit.log("analysis", "error", cid, error=err)
        return {**state, "final_status": "ERROR_NOT_FOUND" if "not found" in err
                else "ERROR_API_UNREACHABLE"}

    facts = {k: contract.get(k) for k in FACT_FIELDS}
    util = contract.get("utilisation_pct")
    print(f"  {contract['vendor']} · {contract['category']} · band {contract['approval_band']} · "
          f"{contract['notice_state']} · util {'n/a' if util is None else f'{util}%'} · "
          f"uplift {contract['proposed_uplift_pct']}% · INR {contract['annual_value_inr']:,}")
    audit.log("analysis", "contract_fetched", cid, facts=facts)

    messages = [{"role": "user",
                 "content": f"Analyse this contract and submit a recommendation:\n"
                            f"{json.dumps(facts, indent=2)}"}]

    for round_no in range(1, MAX_ROUNDS + 1):
        response = get_client().messages.create(
            model=MODEL, max_tokens=MAX_TOKENS, system=ANALYSIS_SYSTEM,
            tools=ANALYSIS_TOOLS, messages=messages,
            extra_body={"output_config": {"effort": EFFORT}},   # works on any SDK version
        )
        messages.append({"role": "assistant", "content": response.content})
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            text = extract_text(response)
            return _no_decision(state, contract,
                                f"Agent ended without a recommendation: {text[:200] or response.stop_reason}")

        results = []
        for block in tool_uses:
            if block.name == "search_policy":
                hits = search_policy(block.input.get("query", ""))
                print(f'  → policy "{block.input.get("query", "")[:50]}": ' +
                      "; ".join(f"{h['source']} §{h['section']} ({h['confidence']:.2f})" for h in hits))
                audit.log("analysis", "policy_search", cid, query=block.input.get("query"),
                          hits=[{k: h[k] for k in ("source", "section", "confidence")} for h in hits])
                result = {"results": hits}
            elif block.name == "submit_recommendation":
                problem = _validate(block.input)
                if problem:
                    print(f"  ✗ submission rejected: {problem}")
                    audit.log("analysis", "submission_rejected", cid, problem=problem)
                    result = {"error": f"Rejected: {problem}. Fix and resubmit."}
                else:
                    a = block.input
                    rec = {"recommendation": a["recommendation"], "confidence": a["confidence"],
                           "rationale": a["rationale"].strip(),
                           "policy_citation": a["policy_citation"].strip(),
                           "estimated_annual_impact_inr": int(a.get("estimated_annual_impact_inr", 0))}
                    print(f"  → {rec['recommendation']} (confidence {rec['confidence']}, "
                          f"round {round_no})")
                    audit.log("analysis", "recommendation", cid, actor=MODEL, rounds=round_no, **rec)
                    return {**state, "contract": contract, **rec}
            else:
                result = {"error": f"Unknown tool {block.name}"}
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": json.dumps(result, default=str)})
        messages.append({"role": "user", "content": results})

    return _no_decision(state, contract, f"No recommendation within MAX_ROUNDS={MAX_ROUNDS}.")


# ======================================================================
# Node 2 - Policy check (pure Python, no model call)
# ======================================================================

def policy_check_node(state: ContractState) -> ContractState:
    print("\n▶ POLICY CHECK")
    contract = state.get("contract") or {}
    reasons: list[str] = []          # accumulate every trigger; do not stop at the first

    band = contract.get("approval_band")
    if band in ("B", "C"):
        reasons.append(f"approval band {band} requires a named human approver")
    if contract.get("notice_state") == "INSIDE_WINDOW":
        reasons.append("contract is INSIDE its notice window (leverage lost)")
    if state.get("recommendation") == "TERMINATE":
        reasons.append("every termination needs written owner confirmation")
    if state.get("confidence") == "LOW":
        reasons.append("analysis confidence is LOW")
    if str(contract.get("owner", "")).strip().upper() == "UNASSIGNED":
        reasons.append("owner is UNASSIGNED")

    required = bool(reasons)
    reason_text = "; ".join(reasons) if reasons else "no policy trigger"
    print(f"  HITL required: {required}")
    for r in reasons:
        print(f"    · {r}")
    audit.log("policy_check", "evaluated", state["contract_id"],
              hitl_required=required, reasons=reasons)
    return {**state, "hitl_required": required, "hitl_reason": reason_text}


def route_after_policy_check(state: ContractState) -> str:
    return "hitl" if state.get("hitl_required") else "report"


# ======================================================================
# Node 3 - Human approval gate
# ======================================================================

def _ask(prompt: str) -> str | None:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return None                   # no human present: fail closed


def hitl_node(state: ContractState) -> ContractState:
    cid, c = state["contract_id"], state.get("contract") or {}
    print("\n▶ HUMAN APPROVAL GATE")
    print(f"  Contract   : {cid}  {c.get('vendor', '')}  (band {c.get('approval_band')}, "
          f"INR {c.get('annual_value_inr', 0):,})")
    print(f"  Proposed   : {state['recommendation']}  ·  confidence {state['confidence']}")
    print(f"  Impact     : INR {state['estimated_annual_impact_inr']:,}/yr")
    print(f"  Policy     : {state['policy_citation']}")
    print(f"  Rationale  : {state['rationale']}")
    print(f"  Gate reason: {state['hitl_reason']}")
    audit.log("hitl", "approval_requested", cid, recommendation=state["recommendation"],
              reasons=state["hitl_reason"])

    answer = None
    while answer not in ("y", "yes", "n", "no"):
        raw = _ask("\n  Approve this recommendation? [y/n]: ")
        if raw is None:
            print("  (no input available: treating as NOT approved)")
            answer = "n"
            break
        answer = raw.lower()

    approved, approver = answer in ("y", "yes"), ""
    if approved:
        while not approver:
            name = _ask("  Approver name: ")
            if name is None:
                approved = False
                print("  (no approver name: treating as NOT approved)")
                break
            approver = name

    print(f"  → {'APPROVED by ' + approver if approved else 'REJECTED: no action will be taken'}")
    audit.log("hitl", "approved" if approved else "rejected", cid,
              actor=approver or "human", recommendation=state["recommendation"])
    return {**state, "hitl_approved": approved, "approver": approver}


# ======================================================================
# Node 4 - Report
# ======================================================================

def report_node(state: ContractState) -> ContractState:
    cid, rec = state["contract_id"], state.get("recommendation", "")
    print("\n▶ REPORT")
    if state.get("final_status", "").startswith("ERROR"):
        final = state["final_status"]
    elif not state.get("hitl_required"):
        final = f"{rec}_AUTO"
    elif state.get("hitl_approved") and rec != NO_DECISION:
        final = f"{rec}_APPROVED"
    else:
        final = "ON_HOLD_REJECTED"
    print(f"  FINAL STATUS: {final}")
    audit.log("report", "final_status", cid, actor=state.get("approver") or "system",
              final_status=final, recommendation=rec, confidence=state.get("confidence"),
              policy_citation=state.get("policy_citation"))
    return {**state, "final_status": final}


# ======================================================================
# Graph
# ======================================================================

def route_after_analysis(state: ContractState) -> str:
    return "report" if state.get("final_status", "").startswith("ERROR") else "policy_check"


def build_graph():
    g = StateGraph(ContractState)
    g.add_node("analysis", analysis_node)
    g.add_node("policy_check", policy_check_node)
    g.add_node("hitl", hitl_node)
    g.add_node("report", report_node)
    g.add_edge(START, "analysis")
    g.add_conditional_edges("analysis", route_after_analysis,
                            {"policy_check": "policy_check", "report": "report"})
    g.add_conditional_edges("policy_check", route_after_policy_check,
                            {"hitl": "hitl", "report": "report"})
    g.add_edge("hitl", "report")
    g.add_edge("report", END)
    return g.compile()


def print_summary(outcomes: list[ContractState]) -> None:
    print(f"\n\n{'=' * 78}\nPORTFOLIO REVIEW\n{'=' * 78}")
    print(f"{'Contract':<10}{'Action':<19}{'Conf':<8}{'Gate':<7}{'Approver':<16}Final status")
    print("-" * 78)
    for o in outcomes:
        print(f"{o['contract_id']:<10}{(o.get('recommendation') or '-'):<19}"
              f"{(o.get('confidence') or '-'):<8}{'HUMAN' if o.get('hitl_required') else 'auto':<7}"
              f"{(o.get('approver') or '-')[:15]:<16}{o.get('final_status', '-')}")


def main() -> None:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key or "your-key" in key:
        raise SystemExit(f"ANTHROPIC_API_KEY missing or still the placeholder in {ROOT / '.env'}")
    if not key.startswith("sk-ant-api") and not os.environ.get("ANTHROPIC_BASE_URL"):
        print(f"  ⚠ Key starts '{key[:11]}', not 'sk-ant-api'. Expect a 401 unless this "
              f"key is meant for a gateway set in ANTHROPIC_BASE_URL.")

    graph = build_graph()
    outcomes = []
    try:
        for cid in (sys.argv[1:] or DEFAULT_PORTFOLIO):
            outcomes.append(graph.invoke(initial_state(cid.upper())))
    except anthropic.AuthenticationError:
        raise SystemExit(f"API key rejected (401). Key in use starts '{key[:12]}...'.")
    except anthropic.NotFoundError as e:
        raise SystemExit(f"Model '{MODEL}' not found: change MODEL at the top of this file. ({e})")
    finally:
        if outcomes:
            print_summary(outcomes)
            audit.summary()


if __name__ == "__main__":
    main()