"""Vera composer: 4-context → WhatsApp message.

Pipeline: extract grounded facts → pick per-trigger-kind playbook → LLM (Groq, temp 0,
JSON mode) → validate (taboos, CTA, send_as, repetition) → retry once → deterministic fallback.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Optional

import requests

def _load_dotenv(path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")) -> None:
    """Minimal .env loader; exported env vars take precedence."""
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip().removeprefix("export ").strip(), v.strip().strip("'\""))


_load_dotenv()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_MODEL") or os.environ.get("LLM_MODEL") or "openai/gpt-oss-20b"
# Groq free-tier limits are per model, so rotate to the next model on a 429 instead of waiting.
LLM_MODELS = [m for m in dict.fromkeys(
    [GROQ_MODEL] + os.environ.get("LLM_FALLBACK_MODELS", "openai/gpt-oss-20b,qwen/qwen3.8-27b,openai/gpt-oss-120b").split(",")) if m]
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "12"))
_cooldown: dict[str, float] = {}  # model -> unix time when it's usable again


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    return json.loads(m.group(0) if m else text)


def llm_json(system: str, user: str, timeout: float = LLM_TIMEOUT,
             budget: float = float(os.environ.get("LLM_BUDGET", "20"))) -> Optional[dict]:
    """JSON-mode call to Groq with model rotation on rate limits. Returns None on failure (caller falls back)."""
    key = os.environ.get("GROQ_API_KEY") or os.environ.get("LLM_API_KEY")
    if not key:
        return None
    deadline = time.time() + budget
    while time.time() < deadline:
        ready = [m for m in LLM_MODELS if _cooldown.get(m, 0) <= time.time()]
        if not ready:  # every model is rate-limited: wait for the first one, if it fits the budget
            wake = min(_cooldown.values())
            if wake >= deadline:
                return None
            time.sleep(max(0.0, wake - time.time()))
            continue
        model = ready[0]
        body = {
            "model": model, "temperature": 0, "seed": 7,
            "max_tokens": 900,  # requested max_tokens counts against the TPM limit
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if "gpt-oss" in model:
            body["reasoning_effort"] = "low"  # reasoning tokens count against max_tokens
        elif "qwen" in model:
            body["reasoning_effort"] = "none"
        try:
            r = requests.post(GROQ_URL, json=body, timeout=min(timeout, max(1.0, deadline - time.time())),
                              headers={"Authorization": f"Bearer {key}"})
            if r.status_code == 429:
                _cooldown[model] = time.time() + float(r.headers.get("retry-after", 5)) + 0.5
                continue
            if r.status_code == 400 and "reasoning_effort" in r.text:  # model doesn't take that param
                _cooldown[model] = time.time() + 3600
                continue
            r.raise_for_status()
            return _parse_json(r.json()["choices"][0]["message"]["content"])
        except Exception:
            _cooldown[model] = time.time() + 10  # transient failure: try another model
    return None


# ---------------------------------------------------------------------------
# Playbook per trigger kind: angle + lever + CTA shape
# ---------------------------------------------------------------------------

PLAYBOOK: dict[str, dict[str, str]] = {
    "research_digest": {"angle": "Share the one digest item most relevant to THIS merchant's patients/customers; cite source + the key number. Offer to pull the abstract and draft a customer-facing WhatsApp.", "lever": "curiosity + reciprocity", "cta": "open_ended"},
    "regulation_change": {"angle": "Compliance heads-up: what changed, the deadline, what the merchant must do. Offer a checklist.", "lever": "loss aversion (deadline)", "cta": "binary_yes_no"},
    "cde_opportunity": {"angle": "Continuing-education / webinar opportunity with date and credits. Offer to register or send details.", "lever": "curiosity + social proof", "cta": "binary_yes_no"},
    "recall_due": {"angle": "Customer-facing recall reminder from the merchant: name the service due, time since last visit, offer the real slots and real catalog price.", "lever": "specificity + effort externalization", "cta": "slot_choice"},
    "customer_lapsed_soft": {"angle": "Customer-facing gentle win-back: reference their last service, a real active offer, easy booking.", "lever": "reciprocity + specificity", "cta": "binary_yes_no"},
    "customer_lapsed_hard": {"angle": "Customer-facing win-back after long gap: warm, no guilt, one concrete reason to return (real offer).", "lever": "reciprocity", "cta": "binary_yes_no"},
    "appointment_tomorrow": {"angle": "Customer-facing appointment reminder for tomorrow with time and service; ask to confirm.", "lever": "commitment", "cta": "binary_confirm"},
    "chronic_refill_due": {"angle": "Customer-facing refill reminder: medicine/plan due date, offer to keep it ready / deliver.", "lever": "effort externalization", "cta": "binary_yes_no"},
    "trial_followup": {"angle": "Follow up after a trial: reference the trial, propose the next concrete step with real price.", "lever": "commitment + specificity", "cta": "binary_yes_no"},
    "wedding_package_followup": {"angle": "Follow up on bridal/wedding package interest with date countdown and real package price.", "lever": "loss aversion (date)", "cta": "binary_yes_no"},
    "winback_eligible": {"angle": "Tell the merchant how many customers are lapsed and offer a drafted win-back campaign using their real offer.", "lever": "loss aversion + effort externalization", "cta": "binary_yes_no"},
    "perf_dip": {"angle": "Name the exact metric drop and window vs baseline, one likely reason grounded in signals, and one fix you'll do for them.", "lever": "loss aversion + effort externalization", "cta": "binary_yes_no"},
    "seasonal_perf_dip": {"angle": "Explain the dip is seasonal (cite seasonal beat), reassure, and propose the counter-move for this season.", "lever": "reciprocity + effort externalization", "cta": "binary_yes_no"},
    "perf_spike": {"angle": "Celebrate the exact spike number, point out how to convert it (post/offer) while attention is high.", "lever": "curiosity + loss aversion (momentum)", "cta": "binary_yes_no"},
    "milestone_reached": {"angle": "Congratulate on the exact milestone, compare to peer stat, suggest using it (post / review ask).", "lever": "social proof", "cta": "binary_yes_no"},
    "review_theme_emerged": {"angle": "Surface the review theme with count and a real quote; propose a concrete response (reply drafts / fix).", "lever": "reciprocity + loss aversion", "cta": "binary_yes_no"},
    "competitor_opened": {"angle": "A new competitor opened nearby (only details given). Contrast with merchant's strengths (rating/reviews) and propose one defensive move.", "lever": "loss aversion + social proof", "cta": "binary_yes_no"},
    "festival_upcoming": {"angle": "Festival in N days: propose a service+price festive offer from the catalog and a drafted post.", "lever": "loss aversion (date) + effort externalization", "cta": "binary_yes_no"},
    "category_seasonal": {"angle": "Seasonal demand shift for the category (cite beat/trend number); propose adjusting offers/posts.", "lever": "curiosity", "cta": "binary_yes_no"},
    "ipl_match_today": {"angle": "IPL match today in their city: suggest a match-time combo from their catalog and a quick post.", "lever": "urgency + effort externalization", "cta": "binary_yes_no"},
    "supply_alert": {"angle": "Supply/recall alert (batch, drug, what to do). Clinical and precise; offer to draft customer notice.", "lever": "loss aversion (risk)", "cta": "binary_yes_no"},
    "gbp_unverified": {"angle": "Google profile is unverified: what they lose (quantify with views/peer), offer 2-min guided verification.", "lever": "loss aversion + effort externalization", "cta": "binary_yes_no"},
    "renewal_due": {"angle": "Plan renewal in N days: show value delivered (their real views/calls), amount, one-tap renew.", "lever": "loss aversion + specificity", "cta": "binary_yes_no"},
    "dormant_with_vera": {"angle": "Re-engage after silence with ONE fresh, specific insight about their account (not a reminder); easy question.", "lever": "curiosity + reciprocity", "cta": "open_ended"},
    "curious_ask_due": {"angle": "Ask the merchant a genuine question about their business this week (e.g. most-asked service), tie it to what you'll do with the answer.", "lever": "asking the merchant", "cta": "open_ended"},
    "active_planning_intent": {"angle": "Merchant previously expressed a plan (see history/payload). Move to action: present the drafted plan with real numbers and ask for go-ahead.", "lever": "effort externalization + commitment", "cta": "binary_yes_no"},
}
DEFAULT_PLAY = {"angle": "Explain why you're messaging now using the trigger payload, anchor on one real number, propose one concrete next step.", "lever": "specificity + reciprocity", "cta": "binary_yes_no"}

CUSTOMER_KINDS = {"recall_due", "customer_lapsed_soft", "customer_lapsed_hard", "appointment_tomorrow",
                  "chronic_refill_due", "trial_followup", "wedding_package_followup"}


# ---------------------------------------------------------------------------
# Fact extraction (only grounded data reaches the LLM)
# ---------------------------------------------------------------------------

def _g(d: Any, *path, default=None):
    for p in path:
        if not isinstance(d, dict):
            return default
        d = d.get(p)
    return default if d is None else d


def wants_hinglish(merchant: dict, customer: Optional[dict]) -> bool:
    if customer:
        pref = str(_g(customer, "identity", "language_pref", default="")).lower()
        return "hi" in pref or "mix" in pref
    return "hi" in (_g(merchant, "identity", "languages", default=[]) or [])


def find_digest_item(category: dict, trigger: dict) -> Optional[dict]:
    payload = trigger.get("payload") or {}
    item_id = payload.get("top_item_id") or payload.get("digest_item_id")
    digest = category.get("digest") or []
    for it in digest:
        if it.get("id") == item_id:
            return it
    if isinstance(payload.get("top_item"), dict):
        return payload["top_item"]
    return None


def extract_facts(category: dict, merchant: dict, trigger: dict, customer: Optional[dict]) -> dict:
    ident = merchant.get("identity") or {}
    perf = merchant.get("performance") or {}
    peer = category.get("peer_stats") or {}
    facts: dict[str, Any] = {
        "merchant": {
            "name": ident.get("name"),
            "owner_first_name": ident.get("owner_first_name"),
            "city": ident.get("city"), "locality": ident.get("locality"),
            "verified": ident.get("verified"),
            "subscription": merchant.get("subscription"),
            "performance_30d": perf,
            "active_offers": [o.get("title") for o in merchant.get("offers") or [] if o.get("status") == "active"],
            "inactive_offers": [f"{o.get('title')} ({o.get('status')})" for o in merchant.get("offers") or [] if o.get("status") != "active"],
            "customer_aggregate": merchant.get("customer_aggregate"),
            "signals": merchant.get("signals"),
            "review_themes": merchant.get("review_themes"),
            "recent_conversation": (merchant.get("conversation_history") or [])[-4:],
        },
        "peer_benchmarks": peer,
        "category": {
            "slug": category.get("slug"),
            "voice": category.get("voice"),
            "offer_catalog": [o.get("title") for o in category.get("offer_catalog") or []][:8],
            "seasonal_beats": category.get("seasonal_beats"),
            "trend_signals": (category.get("trend_signals") or [])[:4],
        },
        "trigger": {k: trigger.get(k) for k in ("id", "kind", "scope", "source", "payload", "urgency", "expires_at")},
    }
    item = find_digest_item(category, trigger)
    if item:
        facts["trigger_digest_item"] = item
    # Content the merchant can reshare, only if the trigger is knowledge-driven
    if trigger.get("kind") in ("research_digest", "curious_ask_due", "dormant_with_vera") and category.get("patient_content_library"):
        facts["shareable_content_titles"] = [c.get("title") for c in category["patient_content_library"]][:3]
    if trigger.get("scope") == "customer" and not customer:
        facts["note"] = ("This is a customer-level event but the customer's profile isn't available. Write to the MERCHANT "
                         "(send_as=vera): tell them what's due for their customer using the trigger payload, and offer to "
                         "send the reminder on their behalf. Don't invent the customer's name or history.")
    if customer:
        facts["customer"] = {k: customer.get(k) for k in ("identity", "relationship", "state", "preferences", "consent")}
        facts["customer"]["identity"] = {k: v for k, v in (customer.get("identity") or {}).items() if k != "phone_redacted"}
    # Pre-computed comparisons (so the LLM doesn't do arithmetic)
    comp = {}
    if perf.get("ctr") is not None and peer.get("avg_ctr"):
        comp["ctr_vs_peer"] = f"{perf['ctr']*100:.1f}% vs peer avg {peer['avg_ctr']*100:.1f}%"
    if perf.get("views") is not None and peer.get("avg_views_30d"):
        comp["views_vs_peer"] = f"{perf['views']} vs peer avg {peer['avg_views_30d']}"
    if perf.get("calls") is not None and peer.get("avg_calls_30d"):
        comp["calls_vs_peer"] = f"{perf['calls']} vs peer avg {peer['avg_calls_30d']}"
    if comp:
        facts["computed"] = comp
    return facts


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM = """You are Vera, magicpin's merchant-growth assistant on WhatsApp for Indian local businesses.
You write ONE WhatsApp message. You are scored 0-10 on each of:
1. Specificity - anchor on concrete verifiable facts from the FACTS JSON (numbers, dates, source citations, service+price like "Cleaning @ ₹299").
2. Category fit - match the category voice/tone; clinical-peer for dentists/pharmacies, warm for salons, energetic for gyms, practical for restaurants. Never use taboo words.
3. Merchant fit - use THIS merchant's own numbers, offers, locality, review themes, recent conversation.
4. Trigger relevance - the first sentence must make clear WHY NOW (the trigger).
5. Engagement - use the given compulsion lever; end with exactly ONE clear, low-effort CTA as the last sentence.

HARD RULES
- If the trigger payload is a placeholder with no details, do NOT invent event specifics (dates, festival names, competitor names, % changes); build the message from the merchant's own data + category facts instead.
- Use ONLY facts in FACTS. Never invent numbers, names, competitors, studies, prices, dates or slots. If something isn't there, don't mention it.
- No preamble ("Hope you're doing well", "I'm reaching out"), no self-introduction beyond "Vera" if needed, no hype/ALL CAPS/"AMAZING".
- Prefer service+price offers from the catalog/active offers over "% off".
- Keep it concise: 2-5 short sentences, ~40-90 words. Max 1 emoji.
- Vera is female: in Hindi use feminine first-person forms ("kar sakti hoon", "bhejti hoon", "karungi"), never "sakta/karunga".
- Language: if HINGLISH=true write natural Hindi-English code-mix in Roman script (e.g. "aapke 78 patients 6 mahine se nahi aaye"); else clear English.
- Merchant-facing: address the owner by first name (Dr. X for dentists). Customer-facing (send_as=merchant_on_behalf): write as the merchant's business to the customer by first name; no medical claims; honour their preferred slots.
- Don't repeat any text from recent_conversation verbatim.
- The rationale must honestly state which facts and lever you used.

Return JSON only:
{"body": str, "cta": "open_ended"|"binary_yes_no"|"binary_confirm"|"slot_choice"|"none", "template_name": str (snake_case, e.g. vera_perf_dip_v1), "template_params": [str, ...] (3-5 key values that fill the template, e.g. name, key fact, offer), "rationale": str (1-2 sentences)}"""


def _prune(x):
    """Drop empty values to save tokens."""
    if isinstance(x, dict):
        return {k: _prune(v) for k, v in x.items() if v not in (None, "", [], {})}
    if isinstance(x, list):
        return [_prune(v) for v in x]
    return x


def build_user_prompt(facts: dict, play: dict, send_as: str, hinglish: bool, feedback: str = "") -> str:
    p = [
        f"TRIGGER KIND: {facts['trigger']['kind']}",
        f"ANGLE: {play['angle']}",
        f"LEVER: {play['lever']}",
        f"PREFERRED CTA: {play['cta']}",
        f"SEND_AS: {send_as}",
        f"HINGLISH: {str(hinglish).lower()}",
        "FACTS:",
        json.dumps(_prune(facts), ensure_ascii=False, default=str, separators=(",", ":")),
    ]
    if feedback:
        p.append(f"\nYOUR PREVIOUS DRAFT FAILED CHECKS: {feedback}. Fix these.")
    return "\n".join(p)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

GENERIC_PHRASES = ["hope you're doing well", "hope you are doing well", "i am reaching out", "i'm reaching out",
                   "amazing deal", "increase your sales"]


def validate(out: dict, category: dict, prior_bodies: list[str]) -> list[str]:
    problems = []
    body = (out.get("body") or "").strip()
    if len(body) < 20:
        problems.append("body empty/too short")
    low = body.lower()
    taboos = _g(category, "voice", "vocab_taboo", default=[]) or _g(category, "voice", "taboos", default=[]) or []
    hit = [t for t in taboos if t and t.lower() in low]
    if hit:
        problems.append(f"contains taboo words {hit}")
    gen = [g for g in GENERIC_PHRASES if g in low]
    if gen:
        problems.append(f"generic phrases {gen}")
    if len(body.split()) > 130:
        problems.append("too long; keep under 90 words")
    if body in prior_bodies:
        problems.append("repeats a previous message verbatim")
    if len(re.findall(r"reply\s+(yes|no|stop|1|2)", low)) > 2:
        problems.append("multiple CTAs")
    return problems


# ---------------------------------------------------------------------------
# Deterministic fallback (no LLM) — still grounded
# ---------------------------------------------------------------------------

def _salutation(merchant: dict, category: dict) -> str:
    first = _g(merchant, "identity", "owner_first_name") or ""
    if category.get("slug") == "dentists" and first:
        return f"Dr. {first}"
    return first or _g(merchant, "identity", "name", default="there")


def fallback_body(category: dict, merchant: dict, trigger: dict, customer: Optional[dict]) -> tuple[str, str]:
    kind = trigger.get("kind", "")
    pl = trigger.get("payload") or {}
    mname = _g(merchant, "identity", "name", default="")
    offers = [o["title"] for o in merchant.get("offers") or [] if o.get("status") == "active"]
    offer = offers[0] if offers else ((category.get("offer_catalog") or [{}])[0].get("title") or "")
    hing = wants_hinglish(merchant, customer)
    if customer:
        cname = _g(customer, "identity", "name", default="")
        slots = pl.get("available_slots") or []
        if slots:
            opts = " ya ".join(s.get("label", "") for s in slots[:2]) if hing else " or ".join(s.get("label", "") for s in slots[:2])
            return (f"Hi {cname}, {mname} here. Aapka {kind.replace('_', ' ')} ho gaya hai — {offer}. Slots: {opts}. Reply 1 or 2 to book." if hing
                    else f"Hi {cname}, {mname} here. Your {kind.replace('_', ' ')} is coming up — {offer}. Open slots: {opts}. Reply 1 or 2 to book."), "slot_choice"
        return (f"Hi {cname}, {mname} here. Kaafi time ho gaya! {offer} aapke liye available hai. Book karein? Reply YES." if hing
                else f"Hi {cname}, {mname} here. It's been a while — {offer} is available for you this week. Reply YES to book."), "binary_yes_no"
    sal = _salutation(merchant, category)
    item = find_digest_item(category, trigger)
    perf = merchant.get("performance") or {}
    if item:
        return (f"{sal}, {item.get('source', 'this week')} — \"{item.get('title')}\". Relevant for your patients/customers in {_g(merchant, 'identity', 'locality', default='your area')}. "
                f"Want me to pull the summary and draft a WhatsApp you can share?"), "open_ended"
    if kind == "perf_dip" and pl.get("delta_pct"):
        pct = abs(pl["delta_pct"]) * 100
        return (f"{sal}, your {pl.get('metric', 'calls')} are down {pct:.0f}% over {pl.get('window', '7d')}. A fresh Google post with '{offer}' usually lifts this fast — I've drafted one. Reply YES to publish."), "binary_yes_no"
    if kind == "renewal_due":
        return (f"{sal}, your {pl.get('plan', '')} plan renews in {pl.get('days_remaining', '')} days (₹{pl.get('renewal_amount', '')}). Last 30 days: {perf.get('views', 0)} views, {perf.get('calls', 0)} calls. Reply YES to renew without a gap."), "binary_yes_no"
    if kind == "curious_ask_due":
        return (f"{sal}, quick one — which service did customers ask about most this week? I'll turn it into a Google post for {_g(merchant, 'identity', 'locality', default='your area')}."), "open_ended"
    peer = category.get("peer_stats") or {}
    bench = f" (peer avg {peer['avg_views_30d']})" if peer.get("avg_views_30d") else ""
    why = {"perf_dip": "calls/views are slipping", "perf_spike": "your listing is getting extra attention right now",
           "festival_upcoming": "the festive rush is about to start", "competitor_opened": "a new competitor listed nearby",
           "milestone_reached": "you just crossed a milestone", "gbp_unverified": "your Google profile is still unverified",
           "ipl_match_today": "there's an IPL match today", "dormant_with_vera": "it's been a while since we spoke",
           "category_seasonal": "seasonal demand is shifting"}.get(kind, "there's a timely opening this week")
    return (f"{sal}, {why} — {mname} had {perf.get('views', 0)} views{bench} and {perf.get('calls', 0)} calls in the last 30 days. "
            f"I've drafted a Google post around '{offer}' to make the most of it. Reply YES to publish."), "binary_yes_no"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compose(category: dict, merchant: dict, trigger: dict, customer: Optional[dict] = None,
            prior_bodies: Optional[list[str]] = None) -> dict:
    prior_bodies = prior_bodies or []
    kind = trigger.get("kind", "")
    is_customer = bool(customer) or trigger.get("scope") == "customer"
    send_as = "merchant_on_behalf" if is_customer and customer else "vera"
    play = PLAYBOOK.get(kind, DEFAULT_PLAY)
    hinglish = wants_hinglish(merchant, customer)
    facts = extract_facts(category, merchant, trigger, customer)

    out, feedback = None, ""
    for _ in range(2):
        cand = llm_json(SYSTEM, build_user_prompt(facts, play, send_as, hinglish, feedback))
        if not cand:
            break
        problems = validate(cand, category, prior_bodies)
        out = cand
        if not problems:
            break
        feedback = "; ".join(problems)

    if not out or not (out.get("body") or "").strip():
        body, cta = fallback_body(category, merchant, trigger, customer)
        out = {"body": body, "cta": cta, "template_name": f"vera_{kind}_v1",
               "template_params": [_salutation(merchant, category), kind],
               "rationale": f"Deterministic fallback for {kind}: anchored on trigger payload + merchant numbers."}

    return {
        "body": out["body"].strip(),
        "cta": out.get("cta") or play["cta"],
        "send_as": send_as,
        "suppression_key": trigger.get("suppression_key") or f"{kind}:{merchant.get('merchant_id')}",
        "rationale": out.get("rationale") or f"{kind} trigger; lever: {play['lever']}",
        "template_name": out.get("template_name") or f"vera_{kind}_v1",
        "template_params": [str(x) for x in (out.get("template_params") or [])][:6],
    }
