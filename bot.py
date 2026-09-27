"""Vera bot — HTTP server implementing the magicpin judge contract.

Run: uvicorn bot:app --host 0.0.0.0 --port 8080
Also exposes `compose(category, merchant, trigger, customer)` per challenge-brief §7.1.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import composer
from conversation_handlers import ConversationState, respond

app = FastAPI(title="Vera bot")


class ForceJSONContentType:
    """Treat every POST /v1/* body as JSON even without a Content-Type header.

    The site's curl examples omit `-H "Content-Type: application/json"`, so curl sends
    form-urlencoded; FastAPI would otherwise reject those bodies with 422.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] == "POST" and scope["path"].startswith("/v1/"):
            headers = [(k, v) for k, v in scope["headers"] if k.lower() != b"content-type"]
            scope = dict(scope, headers=headers + [(b"content-type", b"application/json")])
        await self.app(scope, receive, send)


app.add_middleware(ForceJSONContentType)


@app.exception_handler(RequestValidationError)
async def _invalid_body(request: Request, exc: RequestValidationError):
    details = str(exc.errors()[:3])
    path = request.url.path
    if path == "/v1/tick":  # an empty action list is always a valid answer
        return JSONResponse(status_code=200, content={"actions": []})
    if path == "/v1/context":
        return JSONResponse(status_code=400, content={"accepted": False, "reason": "invalid_payload", "details": details})
    return JSONResponse(status_code=400, content={"error": "invalid_payload", "details": details})
START = time.time()
VALID_SCOPES = {"category", "merchant", "customer", "trigger"}
TICK_BUDGET_S = float(os.environ.get("TICK_BUDGET", "24"))  # judge waits 30s; local simulator only 15s

_lock = threading.Lock()
contexts: dict[tuple[str, str], dict] = {}        # (scope, id) -> {"version", "payload"}
conversations: dict[str, ConversationState] = {}
sent_suppression: set[str] = set()
bodies_by_merchant: dict[str, list[str]] = {}
_pool = ThreadPoolExecutor(max_workers=8)


def compose(category: dict, merchant: dict, trigger: dict, customer: Optional[dict] = None) -> dict:
    out = composer.compose(category, merchant, trigger, customer)
    return {k: out[k] for k in ("body", "cta", "send_as", "suppression_key", "rationale")}


def _get(scope: str, cid: Optional[str]) -> Optional[dict]:
    if not cid:
        return None
    c = contexts.get((scope, cid))
    return c["payload"] if c else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_ts(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/healthz")
async def healthz():
    counts = {s: 0 for s in VALID_SCOPES}
    for (scope, _) in list(contexts):
        counts[scope] += 1
    return {"status": "ok", "uptime_seconds": int(time.time() - START), "contexts_loaded": counts}


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": os.environ.get("TEAM_NAME", "Vera++"),
        "team_members": [m.strip() for m in os.environ.get("TEAM_MEMBERS", "Daksh").split(",") if m.strip()],
        "model": "groq/" + " -> ".join(composer.LLM_MODELS),
        "approach": "Grounded fact extraction + per-trigger-kind playbook (24 kinds) → LLM JSON composer (temp 0) "
                    "→ validator (taboos/CTA/repetition) w/ retry → deterministic fallback. Rule-first reply router "
                    "(auto-reply, opt-out, hostility, intent→action, per-turn language) + LLM for open turns.",
        "contact_email": "penpingos@gmail.com",
        "version": "1.1.0",
        "submitted_at": "2026-09-26T00:00:00Z",
    }


class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: Optional[str] = None


@app.post("/v1/context")
async def push_context(body: CtxBody):
    if body.scope not in VALID_SCOPES:
        return JSONResponse(status_code=400, content={"accepted": False, "reason": "invalid_scope", "details": body.scope})
    key = (body.scope, body.context_id)
    with _lock:
        cur = contexts.get(key)
        if cur and cur["version"] >= body.version:
            if cur["version"] == body.version:  # idempotent re-post
                return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}", "stored_at": _now()}
            return JSONResponse(status_code=409, content={"accepted": False, "reason": "stale_version", "current_version": cur["version"]})
        contexts[key] = {"version": body.version, "payload": body.payload}
    return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}", "stored_at": _now()}


class TickBody(BaseModel):
    now: Optional[str] = None
    available_triggers: list[str] = []


def _build_action(trg_id: str, now: Optional[datetime]) -> Optional[dict]:
    trg = _get("trigger", trg_id)
    if not trg:
        return None
    # The judge's available_triggers already means "active now"; don't second-guess expiry
    # (sim clocks and dataset dates may disagree).
    sup = trg.get("suppression_key") or trg_id
    if sup in sent_suppression:
        return None
    merchant_id = trg.get("merchant_id") or (trg.get("payload") or {}).get("merchant_id")
    merchant = _get("merchant", merchant_id)
    if not merchant:
        return None
    category = _get("category", merchant.get("category_slug")) or {}
    customer_id = trg.get("customer_id") or (trg.get("payload") or {}).get("customer_id")
    customer = _get("customer", customer_id)
    # Customer-scope trigger without customer context: never message the customer blind —
    # ask the merchant (as Vera) to approve the outreach instead.

    prior = bodies_by_merchant.get(merchant_id, [])
    out = composer.compose(category, merchant, trg, customer, prior_bodies=prior)
    conv_id = f"conv_{merchant_id}_{trg_id}"
    return {
        "conversation_id": conv_id, "merchant_id": merchant_id, "customer_id": customer_id if customer else None,
        "send_as": out["send_as"], "trigger_id": trg_id,
        "template_name": out["template_name"], "template_params": out["template_params"],
        "body": out["body"], "cta": out["cta"], "suppression_key": out["suppression_key"],
        "rationale": out["rationale"],
        "_state": ConversationState(conv_id, merchant, category, trg, customer),
    }


@app.post("/v1/tick")
async def tick(body: TickBody):
    now = _parse_ts(body.now)
    # Highest urgency first; dedupe; cap 20.
    trg_ids = [t for t in dict.fromkeys(body.available_triggers)
               if f"conv_{(_get('trigger', t) or {}).get('merchant_id')}_{t}" not in conversations]
    trg_ids.sort(key=lambda t: -((_get("trigger", t) or {}).get("urgency") or 0))
    trg_ids = trg_ids[:20]

    loop = asyncio.get_running_loop()
    futs = [loop.run_in_executor(_pool, _build_action, t, now) for t in trg_ids]
    done, pending = await asyncio.wait(futs, timeout=TICK_BUDGET_S) if futs else (set(), set())
    actions, seen_merchants = [], set()
    for f in futs:
        if f not in done or f.exception() or not f.result():
            continue
        a = f.result()
        key = (a["merchant_id"], a["conversation_id"])
        if key in seen_merchants:  # one action per (merchant, conversation) per tick
            continue
        seen_merchants.add(key)
        state = a.pop("_state")
        state.turns.append({"from": "vera", "body": a["body"]})
        conversations[a["conversation_id"]] = state
        sent_suppression.add(a["suppression_key"])
        bodies_by_merchant.setdefault(a["merchant_id"], []).append(a["body"])
        actions.append(a)
    return {"actions": actions}


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str = "merchant"
    message: str = ""
    received_at: Optional[str] = None
    turn_number: Optional[int] = None


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    state = conversations.get(body.conversation_id)
    if state is None:  # unknown conversation: reconstruct from what we have
        merchant = _get("merchant", body.merchant_id) or {"merchant_id": body.merchant_id, "identity": {}}
        category = _get("category", merchant.get("category_slug")) or {}
        state = ConversationState(body.conversation_id, merchant, category, {"kind": "general"},
                                  _get("customer", body.customer_id))
        conversations[body.conversation_id] = state
    loop = asyncio.get_running_loop()
    try:
        out = await asyncio.wait_for(loop.run_in_executor(_pool, respond, state, body.message), timeout=25)
    except asyncio.TimeoutError:
        out = {"action": "wait", "wait_seconds": 600, "rationale": "Composer timeout; backing off briefly."}
    if out.get("action") == "send":
        state.turns.append({"from": "vera", "body": out["body"]})
    return out


@app.post("/v1/teardown")
async def teardown():
    with _lock:
        contexts.clear(); conversations.clear(); sent_suppression.clear(); bodies_by_merchant.clear()
    return {"ok": True}
