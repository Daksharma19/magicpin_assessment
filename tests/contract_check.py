#!/usr/bin/env python3
"""Endpoint contract check against the website + challenge-testing-brief.md.

Usage: BOT_URL=http://localhost:8080 python tests/contract_check.py
Exits non-zero if any check fails. Stdlib only (like judge_simulator.py).
"""
import json
import os
import sys
import time
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

BOT_URL = os.environ.get("BOT_URL", "http://localhost:8080").rstrip("/")
ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
failures = 0


def call(method, path, body=None, json_header=True, timeout=35):
    """Returns (status, parsed_json_or_text, seconds)."""
    data = None
    headers = {"User-Agent": "vera-contract-check/1.0"}
    if body is not None:
        data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        if json_header:
            headers["Content-Type"] = "application/json"
        else:  # what `curl -d` sends when no -H is given (the website examples)
            headers["Content-Type"] = "application/x-www-form-urlencoded"
    req = urlrequest.Request(BOT_URL + path, data=data, headers=headers, method=method)
    t = time.time()
    try:
        resp = urlrequest.urlopen(req, timeout=timeout)
        status, raw = resp.status, resp.read().decode()
    except urlerror.HTTPError as e:
        status, raw = e.code, e.read().decode()
    dt = time.time() - t
    try:
        return status, json.loads(raw), dt
    except ValueError:
        return status, raw, dt


def check(name, ok, detail=""):
    global failures
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail and not ok else ""))
    if not ok:
        failures += 1


def section(t):
    print(f"\n== {t} ==")


def main():
    print(f"Contract check → {BOT_URL}")

    section("GET /v1/healthz, /v1/metadata")
    st, h, _ = call("GET", "/v1/healthz")
    check("healthz 200 + status ok", st == 200 and h.get("status") == "ok", str(h))
    check("healthz has uptime_seconds", isinstance(h.get("uptime_seconds"), int))
    check("healthz contexts_loaded has 4 scopes",
          set(h.get("contexts_loaded", {})) >= {"category", "merchant", "customer", "trigger"}, str(h))
    st, m, _ = call("GET", "/v1/metadata")
    check("metadata 200 + fields", st == 200 and all(k in m for k in ("team_name", "team_members", "model", "approach", "version")), str(m))

    section("Website curl examples (no Content-Type header)")
    st, r, _ = call("POST", "/v1/context", json.dumps({
        "scope": "merchant", "context_id": "m_site_example", "version": 3,
        "payload": {"identity": {}, "performance": {}, "offers": []},
        "delivered_at": "2026-04-29T10:00:00Z"}), json_header=False)
    check("context without header → 200 accepted", st == 200 and r.get("accepted") is True, f"{st} {r}")
    st, r, _ = call("POST", "/v1/tick", '{ "now": "2026-04-29T10:30:00Z", "available_triggers": ["trg_research_digest_dentists"] }', json_header=False)
    check("tick without header → 200 {actions: []}", st == 200 and r.get("actions") == [], f"{st} {r}")
    st, r, _ = call("POST", "/v1/reply", '{ "conversation_id": "conv_001", "from_role": "merchant", "message": "Yes, send me the abstract", "turn_number": 2 }', json_header=False)
    check("reply without header → 200 valid action", st == 200 and r.get("action") in ("send", "wait", "end"), f"{st} {r}")

    section("POST /v1/context semantics")
    base = {"scope": "category", "context_id": "cc_test", "payload": {"slug": "cc_test"}, "delivered_at": "2026-04-26T10:00:00Z"}
    st, r, _ = call("POST", "/v1/context", {**base, "version": 1})
    check("v1 accepted with ack_id + stored_at", st == 200 and r.get("accepted") and r.get("ack_id") and r.get("stored_at"), str(r))
    st, r, _ = call("POST", "/v1/context", {**base, "version": 1})
    check("same version re-post is a no-op (not an error)", st == 200 and r.get("accepted") is True, f"{st} {r}")
    st, r, _ = call("POST", "/v1/context", {**base, "version": 2})
    check("higher version replaces", st == 200 and r.get("accepted") is True, f"{st} {r}")
    st, r, _ = call("POST", "/v1/context", {**base, "version": 1})
    check("lower version → 409 stale_version + current_version=2",
          st == 409 and r.get("reason") == "stale_version" and r.get("current_version") == 2, f"{st} {r}")
    st, r, _ = call("POST", "/v1/context", {**base, "scope": "planet", "version": 1})
    check("invalid scope → 400 invalid_scope", st == 400 and r.get("reason") == "invalid_scope", f"{st} {r}")
    st, r, _ = call("POST", "/v1/context", "not json at all")
    check("malformed body → 400 (not 422/500)", st == 400, f"{st} {r}")

    section("Base dataset load (warmup)")
    counts = {"category": 0, "merchant": 0, "customer": 0}
    for p in sorted((DATASET / "categories").glob("*.json")):
        st, _, _ = call("POST", "/v1/context", {"scope": "category", "context_id": p.stem, "version": 1,
                                                  "payload": json.loads(p.read_text()), "delivered_at": "x"})
        counts["category"] += st == 200
    seeds = {k: json.loads((DATASET / f"{k}s_seed.json").read_text())[f"{k}s"] for k in ("merchant", "customer", "trigger")}
    for mm in seeds["merchant"]:
        st, _, _ = call("POST", "/v1/context", {"scope": "merchant", "context_id": mm["merchant_id"], "version": 1, "payload": mm})
        counts["merchant"] += st == 200
    for c in seeds["customer"]:
        st, _, _ = call("POST", "/v1/context", {"scope": "customer", "context_id": c["customer_id"], "version": 1, "payload": c})
        counts["customer"] += st == 200
    check("all base contexts accepted",
          counts == {"category": 5, "merchant": len(seeds["merchant"]), "customer": len(seeds["customer"])}, str(counts))
    for t in seeds["trigger"]:
        call("POST", "/v1/context", {"scope": "trigger", "context_id": t["id"], "version": 1, "payload": t})
    st, h, _ = call("GET", "/v1/healthz")
    check("healthz counts reflect pushes", h["contexts_loaded"]["merchant"] >= len(seeds["merchant"]), str(h))

    section("POST /v1/tick")
    trig_ids = [t["id"] for t in seeds["trigger"][:6]]
    st, r, dt = call("POST", "/v1/tick", {"now": "2026-04-26T10:30:00Z", "available_triggers": trig_ids})
    acts = r.get("actions", []) if isinstance(r, dict) else []
    check(f"tick 200 in <30s ({dt:.1f}s)", st == 200 and dt < 30, f"{st}")
    check("≤ 20 actions", len(acts) <= 20)
    check("at least one action for real triggers", len(acts) > 0, str(r)[:200])
    req = ("conversation_id", "merchant_id", "trigger_id", "body", "cta", "send_as", "suppression_key", "rationale")
    for a in acts:
        missing = [k for k in req if k not in a]
        check(f"action {a.get('trigger_id')} has required fields + non-empty body",
              not missing and a.get("body", "").strip() and a.get("send_as") in ("vera", "merchant_on_behalf"), f"missing={missing}")
    check("unique conversation_ids", len({a["conversation_id"] for a in acts}) == len(acts))
    st, r, _ = call("POST", "/v1/tick", {"now": "2026-04-26T10:35:00Z", "available_triggers": []})
    check("empty tick → {actions: []}", st == 200 and r == {"actions": []}, str(r))
    st, r, _ = call("POST", "/v1/tick", {"now": "2026-04-26T10:35:00Z", "available_triggers": trig_ids})
    check("same triggers again → no duplicate sends (suppression)", st == 200 and r.get("actions") == [], str(r)[:200])

    section("POST /v1/reply")
    if acts:
        a = acts[0]
        def reply(msg, n):
            return call("POST", "/v1/reply", {"conversation_id": a["conversation_id"], "merchant_id": a["merchant_id"],
                                              "customer_id": a.get("customer_id"), "from_role": "merchant", "message": msg,
                                              "received_at": "2026-04-26T10:40:00Z", "turn_number": n})
        st, r, dt = reply("Yes, send me the abstract", 2)
        check(f"reply 200 in <30s ({dt:.1f}s)", st == 200 and dt < 30)
        check("action ∈ send/wait/end", r.get("action") in ("send", "wait", "end"), str(r))
        if r.get("action") == "send":
            check("send has non-empty body + cta", bool(r.get("body", "").strip()) and "cta" in r, str(r))
            st, r2, _ = reply("ok", 3)
            check("no verbatim repeat of previous body", r2.get("body") != r.get("body"), str(r2))
        st, r, _ = reply("busy right now, call me later", 4)
        check("'later' → wait with wait_seconds", r.get("action") == "wait" and isinstance(r.get("wait_seconds"), int), str(r))
    st, r, _ = call("POST", "/v1/reply", {"conversation_id": "conv_never_seen", "merchant_id": "m_unknown",
                                          "from_role": "merchant", "message": "hello?", "turn_number": 1})
    check("unknown conversation handled (200, valid action)", st == 200 and r.get("action") in ("send", "wait", "end"), f"{st} {r}")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
