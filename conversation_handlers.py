"""Multi-turn reply handling: deterministic routing first, LLM for open conversation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from composer import llm_json, wants_hinglish, _salutation


@dataclass
class ConversationState:
    conversation_id: str
    merchant: dict
    category: dict
    trigger: dict
    customer: Optional[dict] = None
    turns: list = field(
        default_factory=list
    )  # [{"from": "vera"|"merchant"|"customer", "body": str}]
    auto_reply_count: int = 0
    mode: str = "pitch"  # pitch | action | ended
    ended: bool = False

    def bot_bodies(self):
        return [t["body"] for t in self.turns if t["from"] == "vera"]


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

AUTO_REPLY_PATTERNS = [
    r"thank(s| you) for (contacting|reaching|your message)",
    r"we will (get back|revert|respond)",
    r"will get back to you",
    r"our team will",
    r"automated (assistant|message|reply)",
    r"currently (unavailable|closed|away)",
    r"business hours",
    r"jaankari ke liye.*shukriya",
    r"team tak pahuncha",
    r"this is an auto",
    r"auto[- ]?reply",
    r"out of office",
]
OPT_OUT = r"\b(stop messaging|stop sending|stop it|^stop$|unsubscribe|not interested|no thanks|don'?t (message|contact)|band karo|mat bhejo|nahi chahiye|interested nahi|remove me)\b"
ABUSE = r"\b(idiot|stupid|fuck|shit|bakwas|chutiya|bewakoof|scam|fraud|spam|harass|bloody|nonsense)\b"
INTENT_YES = (
    r"\b(yes|yess|yeah|yep|ok(ay)?|sure|go ahead|let'?s do it|do it|proceed|haan|ha+n?ji|han|theek hai|thik hai|"
    r"kar do|karo|chalo|done|please do|sounds good|send (it|me)|i want to join|join karna|judna|start)\b"
)
LATER = r"\b(later|busy|baad mein|kal|tomorrow|abhi nahi|not now|call me later|in a meeting)\b"
OFF_TOPIC = r"\b(gst|income tax|itr|loan|visa|passport|aadhaar|electricity bill|legal notice|court)\b"
# Auto-reply counts per merchant: the judge may use a fresh conversation_id each turn.
AUTO_BY_MERCHANT: dict[str, int] = {}
LAST_MSGS_BY_MERCHANT: dict[str, list[str]] = {}

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
HINDI_ROMAN = r"\b(hai|haan|nahi|kya|karo|kaise|mujhe|aap|mera|kyun|chahiye|accha|theek|bhai|ji)\b"


def is_auto_reply(msg: str, state: ConversationState) -> bool:
    low = msg.lower()
    if any(re.search(p, low) for p in AUTO_REPLY_PATTERNS):
        return True
    mid = state.merchant.get("merchant_id") or ""
    prev = [
        t["body"].strip().lower() for t in state.turns if t["from"] != "vera"
    ] + LAST_MSGS_BY_MERCHANT.get(mid, [])
    return prev.count(low.strip()) >= 2  # same text verbatim 3+ times


def reply_lang_hinglish(msg: str, state: ConversationState) -> bool:
    if DEVANAGARI.search(msg) or re.search(HINDI_ROMAN, msg.lower()):
        return True
    if re.search(r"[a-z]{3,}", msg.lower()) and len(msg.split()) >= 4:
        return False  # clear English sentence → switch to English
    return wants_hinglish(state.merchant, state.customer)


# ---------------------------------------------------------------------------
# Respond
# ---------------------------------------------------------------------------

REPLY_SYSTEM = """You are Vera, magicpin's merchant assistant on WhatsApp, continuing an ongoing conversation.
Rules:
- MODE=action means the merchant has agreed: start with 'Done —' or 'Sending now —', confirm exactly what you are doing now (concrete, using facts), state what they'll get and when, ask at most one tiny confirmation. NEVER ask qualifying questions in action mode.
- Answer merchant questions directly using ONLY the FACTS; if you don't know, say so honestly and offer what you can do.
- Off-topic requests (GST, loans, etc.): one polite line that it's outside what you do, then steer back to the open thread.
- Never re-introduce yourself, no preamble, never repeat a previous bot message. 1-4 short sentences. One CTA at most.
- Vera is female: in Hindi use feminine first-person forms ("kar sakti hoon", "bhejti hoon", "karungi"), never "sakta/karunga".
- Language: HINGLISH=true → natural Roman Hindi-English mix; else English.
- Never invent numbers, names, prices or studies.
Return JSON: {"body": str, "cta": "open_ended"|"binary_yes_no"|"none", "rationale": str}"""


def _facts_brief(state: ConversationState) -> dict:
    m = state.merchant
    return {
        "merchant": {
            "name": m.get("identity", {}).get("name"),
            "locality": m.get("identity", {}).get("locality"),
            "performance": m.get("performance"),
            "active_offers": [
                o.get("title")
                for o in m.get("offers") or []
                if o.get("status") == "active"
            ],
            "signals": m.get("signals"),
            "customer_aggregate": m.get("customer_aggregate"),
        },
        "category_offers": [
            o.get("title") for o in state.category.get("offer_catalog") or []
        ][:6],
        "trigger": {
            "kind": state.trigger.get("kind"),
            "payload": state.trigger.get("payload"),
        },
        "customer": (
            {"name": (state.customer or {}).get("identity", {}).get("name")}
            if state.customer
            else None
        ),
    }


def _send(body, cta, rationale):
    return {"action": "send", "body": body, "cta": cta, "rationale": rationale}


def respond(state: ConversationState, merchant_message: str) -> dict:
    msg = (merchant_message or "").strip()
    low = msg.lower()
    hing = reply_lang_hinglish(msg, state)
    is_auto = is_auto_reply(msg, state)
    mid = state.merchant.get("merchant_id") or ""
    LAST_MSGS_BY_MERCHANT.setdefault(mid, []).append(low)
    state.turns.append(
        {"from": "customer" if state.customer else "merchant", "body": msg}
    )

    if state.ended:
        # Reopen only if a real merchant comes back with intent or a question.
        if (
            is_auto
            or re.search(OPT_OUT, low)
            or not (re.search(INTENT_YES, low) or "?" in msg)
        ):
            return {"action": "end", "rationale": "Conversation already closed."}
        state.ended = False

    # 1. Auto-reply: one attempt to reach the owner, then exit.
    if is_auto:
        AUTO_BY_MERCHANT[mid] = AUTO_BY_MERCHANT.get(mid, 0) + 1
        state.auto_reply_count = max(state.auto_reply_count + 1, AUTO_BY_MERCHANT[mid])
        if state.auto_reply_count == 1:
            body = (
                "Lagta hai yeh auto-reply hai 🙂 Jab owner/manager free hon, bas 'YES' bhej dein — main 2 minute mein poora update share kar dungi."
                if hing
                else "Looks like an auto-reply. Whenever the owner/manager is free, just reply YES and I'll share the full update in 2 minutes."
            )
            return _send(
                body,
                "binary_yes_no",
                "Detected WhatsApp Business auto-reply; single low-effort nudge aimed at the owner.",
            )
        state.ended = True
        return {
            "action": "end",
            "rationale": "Canned auto-reply again after one owner-directed nudge; exiting gracefully instead of burning turns.",
        }

    # 2. Explicit opt-out wins (even if hostile), then de-escalate plain hostility once.
    if re.search(OPT_OUT, low):
        state.ended = True
        return {
            "action": "end",
            "rationale": "Merchant opted out / not interested; respecting it and closing.",
        }

    if re.search(ABUSE, low) and not re.search(OFF_TOPIC, low):
        if any(t.get("abuse") for t in state.turns[:-1]):
            state.ended = True
            return {
                "action": "end",
                "rationale": "Repeated hostility; exiting politely.",
            }
        state.turns[-1]["abuse"] = True
        body = (
            "Maafi chahti hoon agar message pareshaan kar gaya. Main sirf aapke Google profile aur customers badhane mein help karti hoon — zaroorat ho toh 'YES' bhej dijiye, warna main aur message nahi karungi."
            if hing
            else "Sorry if this felt intrusive. I only help with your Google profile and bringing in customers — reply YES if that's useful, otherwise I won't message again."
        )
        return _send(
            body,
            "binary_yes_no",
            "Hostile reply: de-escalate once, restate mission, offer clean exit.",
        )

    # 3. Later / busy.
    if re.search(LATER, low) and not re.search(INTENT_YES, low):
        return {
            "action": "wait",
            "wait_seconds": (
                3600 if "kal" not in low and "tomorrow" not in low else 86400
            ),
            "rationale": "Merchant asked for time; backing off.",
        }

    # 4. Intent → action mode.
    if re.search(INTENT_YES, low) and not re.search(OFF_TOPIC, low):
        state.mode = "action"

    # 5. Too many turns without progress → graceful close.
    vera_turns = len(state.bot_bodies())
    if vera_turns >= 5:
        state.ended = True
        return {
            "action": "end",
            "rationale": "Reached turn budget; closing to avoid fatigue.",
        }

    # 6. LLM for the actual content.
    history = "\n".join(f"[{t['from'].upper()}] {t['body']}" for t in state.turns[-8:])
    user = "\n".join(
        [
            f"MODE: {state.mode}",
            f"HINGLISH: {str(hing).lower()}",
            f"OFF_TOPIC_DETECTED: {bool(re.search(OFF_TOPIC, low))}",
            "FACTS: "
            + json.dumps(_facts_brief(state), ensure_ascii=False, default=str),
            "CONVERSATION:",
            history,
            "Write Vera's next message.",
        ]
    )
    out = llm_json(REPLY_SYSTEM, user)
    if out and out.get("body") and out["body"].strip() not in state.bot_bodies():
        return _send(
            out["body"].strip(),
            out.get("cta", "open_ended"),
            out.get("rationale", f"mode={state.mode}"),
        )

    # Fallbacks without LLM.
    sal = _salutation(state.merchant, state.category)
    if re.search(OFF_TOPIC, low):
        body = (
            f"{sal}, GST/tax mein main help nahi kar paungi — woh aapke CA ka kaam hai. Par aapka {state.trigger.get('kind', '').replace('_', ' ')} wala kaam main abhi kar sakti hoon — karein?"
            if hing
            else f"{sal}, tax/GST filing is outside what I can do — your CA is best for that. I can still finish the {state.trigger.get('kind', '').replace('_', ' ')} piece for you now — shall I?"
        )
        return _send(
            body,
            "binary_yes_no",
            "Off-topic request declined politely; steered back to mission.",
        )
    if state.mode == "action":
        body = (
            "Done — main abhi draft taiyaar kar rahi hoon, 10 minute mein aapko preview bhejti hoon. Aap bas approve kar dena."
            if hing
            else "On it — I'm preparing the draft now and will send you a preview within 10 minutes for a one-tap approval."
        )
        return _send(
            body,
            "none",
            "Merchant committed; switched to action mode with a concrete next step, no qualifying questions.",
        )
    body = (
        "Samajh gayi. Kya main aapke liye ek ready draft bana doon jise aap bas approve karein?"
        if hing
        else "Got it. Want me to put together a ready-to-approve draft for you?"
    )
    return _send(body, "binary_yes_no", "Keep momentum with a single low-effort offer.")
